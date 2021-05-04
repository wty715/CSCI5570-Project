#pragma once

#include "alex_base.h"

namespace alex {
typedef std::pair<double, double> Value;
    class AlexNode {
    public:
    
    // Whether this node is a leaf (data) node
    bool is_leaf_ = false;
    // Power of 2 to which the pointer to this node is duplicated in its parent model node
    // For example, if duplication_factor_ is 3, then there are 8 redundant pointers to this node in its parent
    uint8_t duplication_factor_ = 0;
    // Node's level in the RMI. Root node is level 0
    short level_ = 0;
    // Both model nodes and data nodes nodes use models
    LinearModel<double> model_;
    // Could be either the expected or empirical cost, depending on how this field is used
    double cost_ = 0.0;
    AlexNode() = default;
    AlexNode(short level) : level_(level) {}
    AlexNode(short level, bool is_leaf) : level_(level), is_leaf_(is_leaf) {}
    virtual ~AlexNode() = default;
    // The size in bytes of all member variables in this class
    virtual long long node_size() const = 0;
};

class AlexModelNode : public AlexNode {
    public:

    typedef typename std::allocator<Value>::template rebind<AlexModelNode>::other alloc_type;
    typedef typename std::allocator<Value>::template rebind<AlexNode*>::other pointer_alloc_type;
    const std::allocator<Value>& allocator_;
    // Number of logical children. Must be a power of 2
    int num_children_ = 0;
    // Array of pointers to children
    AlexNode** children_ = nullptr;

    AlexModelNode(const std::allocator<Value>& alloc = std::allocator<Value>())
        : AlexNode(0, false), allocator_(alloc) {}

    AlexModelNode(short level, const std::allocator<Value>& alloc = std::allocator<Value>())
        : AlexNode(level, false), allocator_(alloc) {}
    
    ~AlexModelNode() {
        if (children_ == nullptr) {
            return;
        }
        pointer_alloc_type(allocator_).deallocate(children_, num_children_);
    }

    AlexModelNode(const AlexModelNode& other) : AlexNode(other), allocator_(other.allocator_), num_children_(other.num_children_) {
        children_ = new (pointer_alloc_type(allocator_).allocate(other.num_children_)) AlexNode*[other.num_children_];
        std::copy(other.children_, other.children_ + other.num_children_, children_);
    }
    // Given a key, traverses to the child node responsible for that key
    inline AlexNode* get_child_node(const double& key) {
        int bucketID = this->model_.predict(key);
        bucketID = std::min<int>(std::max<int>(bucketID, 0), num_children_ - 1);
        return children_[bucketID];
    }
    // Expand by a power of 2 by creating duplicates of all existing child pointers.
    // Input is the base 2 log of the expansion factor, in order to guarantee expanding by a power of 2.
    // Returns the expansion factor.
    int expand(int log2_expansion_factor) {
        int expansion_factor = 1 << log2_expansion_factor;
        int num_new_children = num_children_ * expansion_factor;
        auto new_children = new (pointer_alloc_type(allocator_).allocate(num_new_children)) AlexNode*[num_new_children];
        int cur = 0;
        while (cur < num_children_) {
            AlexNode* cur_child = children_[cur];
            int cur_child_repeats = 1 << cur_child->duplication_factor_;
            for (int i = expansion_factor * cur; i < expansion_factor * (cur + cur_child_repeats); i++) {
                new_children[i] = cur_child;
            }
            cur_child->duplication_factor_ += log2_expansion_factor;
            cur += cur_child_repeats;
        }
        pointer_alloc_type(allocator_).deallocate(children_, num_children_);
        children_ = new_children;
        num_children_ = num_new_children;
        this->model_.expand(expansion_factor);
        return expansion_factor;
    }
    long long node_size() const override {
      long long size = sizeof(AlexModelNode);
      size += num_children_ * sizeof(AlexNode*);  // pointers to children
      return size;
    }
};

class AlexDataNode : public AlexNode {
    public:

    typedef typename std::allocator<Value>::template rebind<AlexDataNode>::other alloc_type;
    typedef typename std::allocator<Value>::template rebind<double>::other key_alloc_type;
    typedef typename std::allocator<Value>::template rebind<double>::other payload_alloc_type;
    typedef typename std::allocator<Value>::template rebind<uint64_t>::other bitmap_alloc_type;

    const AlexCompare& key_less_;
    const std::allocator<Value>& allocator_;

    AlexDataNode* next_leaf_ = nullptr;
    AlexDataNode* prev_leaf_ = nullptr;

    // We store key and payload arrays separately in data nodes
    double* key_slots_ = nullptr;      // holds keys
    double* payload_slots_ = nullptr;  // holds payloads, must be same size as key_slots

    int data_capacity_ = 0;   // size of key/data_slots array
    int num_keys_ = 0;        // number of filled key/data slots (as opposed to gaps)

    // Bitmap: each uint64_t represents 64 positions in reverse order
    // (i.e., each uint64_t is "read" from the right-most bit to the left-most bit)
    uint64_t* bitmap_ = nullptr;
    int bitmap_size_ = 0;  // number of int64_t in bitmap

    // Variables related to resizing (expansions and contractions)
    static constexpr double kMaxDensity_ = 0.8;   // density after contracting, also determines the expansion threshold
    static constexpr double kInitDensity_ = 0.7;  // density of data nodes after bulk loading
    static constexpr double kMinDensity_ = 0.6;   // density after expanding, also determines the contraction threshold
    double expansion_threshold_ = 1;              // expand after num_keys is >= this number
    double contraction_threshold_ = 0;            // contract after num_keys is < this number
    static constexpr int kDefaultMaxDataNodeBytes_ = 1 << 24;     // by default, maximum data node size is 16MB
    int max_slots_ = kDefaultMaxDataNodeBytes_ / sizeof(Value);   // cannot expand beyond this number of key/data slots

    // Counters used in cost models, does not reset after resizing
    long long num_shifts_ = 0, num_exp_search_iterations_ = 0;
    int num_lookups_ = 0, num_inserts_ = 0, num_resizes_ = 0; // technically not required, but nice to have

    // Variables for determining append-mostly behavior
    double max_key_ = std::numeric_limits<double>::lowest();  // max key in node, updates after inserts but not erases
    double min_key_ = std::numeric_limits<double>::max();     // min key in node, updates after inserts but not erases
    int num_right_out_of_bounds_inserts_ = 0;                 // number of inserts that are larger than the max key
    int num_left_out_of_bounds_inserts_ = 0;                  // number of inserts that are smaller than the min key
    // Node is considered append-mostly if the fraction of inserts that are out of bounds is above this threshold
    // Append-mostly nodes will expand in a manner that anticipates further appends
    static constexpr double kAppendMostlyThreshold = 0.9;

    // Purely for benchmark debugging purposes
    double expected_avg_exp_search_iterations_ = 0;
    double expected_avg_shifts_ = 0;

    // Placed at the end of the key/data slots if there are gaps after the max key
    static constexpr double kEndSentinel_ = std::numeric_limits<double>::max();

    /*** Constructors and destructors ***/
    AlexDataNode(const AlexCompare& comp = AlexCompare(), const std::allocator<Value>& alloc = std::allocator<Value>())
        : AlexNode(0, true), key_less_(comp), allocator_(alloc) {}

    AlexDataNode(short level, int max_data_node_slots,
                 const AlexCompare& comp = AlexCompare(), const std::allocator<Value>& alloc = std::allocator<Value>())
        : AlexNode(level, true), key_less_(comp), allocator_(alloc), max_slots_(max_data_node_slots) {}

    ~AlexDataNode() {
        if (key_slots_ == nullptr) {
            return;
        }
        key_alloc_type(allocator_).deallocate(key_slots_, data_capacity_);
        payload_alloc_type(allocator_).deallocate(payload_slots_, data_capacity_);
        bitmap_alloc_type(allocator_).deallocate(bitmap_, bitmap_size_);
    }

    AlexDataNode(const AlexDataNode& other)
        : AlexNode(other),
          key_less_(other.key_less_),
          allocator_(other.allocator_),
          next_leaf_(other.next_leaf_),
          prev_leaf_(other.prev_leaf_),
          data_capacity_(other.data_capacity_),
          num_keys_(other.num_keys_),
          bitmap_size_(other.bitmap_size_),
          expansion_threshold_(other.expansion_threshold_),
          contraction_threshold_(other.contraction_threshold_),
          max_slots_(other.max_slots_),
          num_shifts_(other.num_shifts_),
          num_exp_search_iterations_(other.num_exp_search_iterations_),
          num_lookups_(other.num_lookups_),
          num_inserts_(other.num_inserts_),
          num_resizes_(other.num_resizes_),
          max_key_(other.max_key_),
          min_key_(other.min_key_),
          num_right_out_of_bounds_inserts_(other.num_right_out_of_bounds_inserts_),
          num_left_out_of_bounds_inserts_(other.num_left_out_of_bounds_inserts_),
          expected_avg_exp_search_iterations_(other.expected_avg_exp_search_iterations_),
          expected_avg_shifts_(other.expected_avg_shifts_) {

        // Separate keys and payloads
        key_slots_ = new (key_alloc_type(allocator_).allocate(other.data_capacity_)) double[other.data_capacity_];
        std::copy(other.key_slots_, other.key_slots_ + other.data_capacity_, key_slots_);
        payload_slots_ = new (payload_alloc_type(allocator_).allocate(other.data_capacity_)) double[other.data_capacity_];
        std::copy(other.payload_slots_, other.payload_slots_ + other.data_capacity_, payload_slots_);
        bitmap_ = new (bitmap_alloc_type(allocator_).allocate(other.bitmap_size_)) uint64_t[other.bitmap_size_];
        std::copy(other.bitmap_, other.bitmap_ + other.bitmap_size_, bitmap_);
    }

    /*** General helper functions ***/
    inline double& get_key(int pos) const {
        return key_slots_[pos];
    }
    inline double& get_payload(int pos) const {
        return payload_slots_[pos];
    }
    // ------------------get functions end------------------
    // Check whether the position corresponds to a key (as opposed to a gap)
    inline bool check_exists(int pos) const {
        assert(pos >= 0 && pos < data_capacity_);
        int bitmap_pos = pos >> 6;
        int bit_pos = pos - (bitmap_pos << 6);
        return static_cast<bool>(bitmap_[bitmap_pos] & (1ULL << bit_pos));
    }
    // Mark the entry for position in the bitmap
    inline void set_bit(int pos) {
        assert(pos >= 0 && pos < data_capacity_);
        int bitmap_pos = pos >> 6;
        int bit_pos = pos - (bitmap_pos << 6);
        bitmap_[bitmap_pos] |= (1ULL << bit_pos);
    }
    // Mark the entry for position in the bitmap
    inline void set_bit(uint64_t bitmap[], int pos) {
        int bitmap_pos = pos >> 6;
        int bit_pos = pos - (bitmap_pos << 6);
        bitmap[bitmap_pos] |= (1ULL << bit_pos);
    }
    // Unmark the entry for position in the bitmap
    inline void unset_bit(int pos) {
        assert(pos >= 0 && pos < data_capacity_);
        int bitmap_pos = pos >> 6;
        int bit_pos = pos - (bitmap_pos << 6);
        bitmap_[bitmap_pos] &= ~(1ULL << bit_pos);
    }
    // ------------------bitmap functions end------------------
    // Value of first (i.e., min) key
    double first_key() const {
        for (int i = 0; i < data_capacity_; i++) {
            if (check_exists(i)) return get_key(i);
        }
        return std::numeric_limits<double>::max();
    }
    // Value of last (i.e., max) key
    double last_key() const {
        for (int i = data_capacity_ - 1; i >= 0; i--) {
            if (check_exists(i)) return get_key(i);
        }
        return std::numeric_limits<double>::lowest();
    }
    // Position in key/data_slots of first (i.e., min) key
    int first_pos() const {
        for (int i = 0; i < data_capacity_; i++) {
            if (check_exists(i)) return i;
        }
        return 0;
    }
    // Position in key/data_slots of last (i.e., max) key
    int last_pos() const {
        for (int i = data_capacity_ - 1; i >= 0; i--) {
            if (check_exists(i)) return i;
        }
        return 0;
    }
    // Number of keys between positions left and right (exclusive) in key/data_slots
    int num_keys_in_range(int left, int right) const {
        assert(left >= 0 && left < right && right <= data_capacity_);
        int num_keys = 0;
        int left_bitmap_idx = left >> 6;
        int right_bitmap_idx = right >> 6;
        if (left_bitmap_idx == right_bitmap_idx) {
            uint64_t bitmap_data = bitmap_[left_bitmap_idx];
            int left_bit_pos = left - (left_bitmap_idx << 6);
            bitmap_data &= ~((1ULL << left_bit_pos) - 1);
            int right_bit_pos = right - (right_bitmap_idx << 6);
            bitmap_data &= ((1ULL << right_bit_pos) - 1);
            num_keys += _mm_popcnt_u64(bitmap_data);
        }
        else {
            uint64_t left_bitmap_data = bitmap_[left_bitmap_idx];
            int bit_pos = left - (left_bitmap_idx << 6);
            left_bitmap_data &= ~((1ULL << bit_pos) - 1);
            num_keys += _mm_popcnt_u64(left_bitmap_data);
            for (int i = left_bitmap_idx + 1; i < right_bitmap_idx; i++) {
                num_keys += _mm_popcnt_u64(bitmap_[i]);
            }
            if (right_bitmap_idx != bitmap_size_) {
                uint64_t right_bitmap_data = bitmap_[right_bitmap_idx];
                bit_pos = right - (right_bitmap_idx << 6);
                right_bitmap_data &= ((1ULL << bit_pos) - 1);
                num_keys += _mm_popcnt_u64(right_bitmap_data);
            }
        }
        return num_keys;
    }
    // ------------------key functions end------------------
    // True if a < b
    inline bool key_less(const double& a, const double& b) const {
        return key_less_(a, b);
    }
    // True if a <= b
    inline bool key_lessequal(const double& a, const double& b) const {
        return !key_less_(b, a);
    }
    // True if a > b
    inline bool key_greater(const double& a, const double& b) const {
        return key_less_(b, a);
    }
    // True if a >= b
    inline bool key_greaterequal(const double& a, const double& b) const {
        return !key_less_(a, b);
    }
    // True if a == b
    inline bool key_equal(const double& a, const double& b) const {
        return !key_less_(a, b) && !key_less_(b, a);
    }
    // ------------------comparison functions end------------------
    /*** Iterator ***/

    // Forward iterator meant for iterating over a single data node, which is a "normal" non-const iterator.
    class Iterator {
        public:

        AlexDataNode* node_;
        int cur_idx_ = 0;               // current position in key/data_slots, -1 if at end
        int cur_bitmap_idx_ = 0;        // current position in bitmap
        uint64_t cur_bitmap_data_ = 0;  // caches the relevant data in the current bitmap position

        Iterator(AlexDataNode* node) : node_(node) {}
        Iterator(AlexDataNode* node, int idx) : node_(node), cur_idx_(idx) {
            cur_bitmap_idx_ = cur_idx_ >> 6;
            cur_bitmap_data_ = node_->bitmap_[cur_bitmap_idx_];

            // Zero out extra bits
            int bit_pos = cur_idx_ - (cur_bitmap_idx_ << 6);
            cur_bitmap_data_ &= ~((1ULL << bit_pos) - 1);

            (*this)++;
        }

        void operator++(int) {
            while (cur_bitmap_data_ == 0) {
                cur_bitmap_idx_++;
                if (cur_bitmap_idx_ >= node_->bitmap_size_) {
                    cur_idx_ = -1;
                    return;
                }
                cur_bitmap_data_ = node_->bitmap_[cur_bitmap_idx_];
            }
            uint64_t bit = extract_rightmost_one(cur_bitmap_data_);
            cur_idx_ = get_offset(cur_bitmap_idx_, bit);
            cur_bitmap_data_ = remove_rightmost_one(cur_bitmap_data_);
        }
        Value operator*() const {
            return std::make_pair(node_->key_slots_[cur_idx_], node_->payload_slots_[cur_idx_]);
        }
        bool operator==(const Iterator& rhs) const {
            return cur_idx_ == rhs.cur_idx_;
        }
        bool operator!=(const Iterator& rhs) const {
            return !(*this == rhs);
        }

        const double& key() const {
          return node_->key_slots_[cur_idx_];
        }
        double& payload() const {
          return node_->payload_slots_[cur_idx_];
        }
        bool is_end() const {
            return cur_idx_ == -1;
        }
    };
    Iterator begin() {
        return Iterator(this, 0);
    }

    /*** Cost model ***/
    // Empirical average number of shifts per insert
    double shifts_per_insert() const {
        if (num_inserts_ == 0) {
            return 0;
        }
        return num_shifts_ / static_cast<double>(num_inserts_);
    }
    // Empirical average number of exponential search iterations per operation (either lookup or insert)
    double exp_search_iterations_per_operation() const {
        if (num_inserts_ + num_lookups_ == 0) {
            return 0;
        }
        return num_exp_search_iterations_ / static_cast<double>(num_inserts_ + num_lookups_);
    }
    // Cost calculation
    double empirical_cost() const {
        if (num_inserts_ + num_lookups_ == 0) {
            return 0;
        }
        double frac_inserts = static_cast<double>(num_inserts_) / (num_inserts_ + num_lookups_);
        return kExpSearchIterationsWeight * exp_search_iterations_per_operation()
             + kShiftsWeight * shifts_per_insert() * frac_inserts;
    }
    // Empirical fraction of operations (either lookup or insert) that are inserts
    double frac_inserts() const {
        int num_ops = num_inserts_ + num_lookups_;
        if (num_ops == 0) {
            return 0;  // if no operations, assume no inserts
        }
        return static_cast<double>(num_inserts_) / (num_inserts_ + num_lookups_);
    }
    void reset_stats() {
        num_shifts_ = 0;
        num_exp_search_iterations_ = 0;
        num_lookups_ = 0;
        num_inserts_ = 0;
        num_resizes_ = 0;
    }
    // Computes the expected cost of the current node
    double compute_expected_cost(double frac_inserts = 0) {
        if (num_keys_ == 0) {
            return 0;
        }
        ExpectedSearchIterationsAccumulator search_iters_accumulator;
        ExpectedShiftsAccumulator shifts_accumulator(data_capacity_);
        for (Iterator it(this, 0); !it.is_end(); it++) {
            int predicted_position = std::max(0, std::min(data_capacity_ - 1, this->model_.predict(it.key())));
            search_iters_accumulator.accumulate(it.cur_idx_, predicted_position);
            shifts_accumulator.accumulate(it.cur_idx_, predicted_position);
        }
        expected_avg_exp_search_iterations_ = search_iters_accumulator.get_stat();
        expected_avg_shifts_ = shifts_accumulator.get_stat();
        double cost = kExpSearchIterationsWeight * expected_avg_exp_search_iterations_
                    + kShiftsWeight * expected_avg_shifts_ * frac_inserts;
        return cost;
    }
    // Computes the expected cost of a data node constructed using the input dense array of keys
    // Assumes existing_model is trained on the dense array of keys
    static double compute_expected_cost(const Value* values, int num_keys, double density, double expected_insert_frac,
                                        const LinearModel<double>* existing_model = nullptr, DataNodeStats* stats = nullptr) {
        if (num_keys == 0) {
            return 0;
        }
        int data_capacity = std::max(static_cast<int>(num_keys / density), num_keys + 1);

        // Compute what the node's model would be
        LinearModel<double> model;
        if (existing_model == nullptr) {
            build_model(values, num_keys, &model);
        }
        else {
            model.a_ = existing_model->a_;
            model.b_ = existing_model->b_;
        }
        model.expand(static_cast<double>(data_capacity) / num_keys);

        // Compute expected stats in order to compute the expected cost
        double cost = 0;
        double expected_avg_exp_search_iterations = 0;
        double expected_avg_shifts = 0;
        if (expected_insert_frac == 0) {
            ExpectedSearchIterationsAccumulator acc;
            build_node_implicit(values, num_keys, data_capacity, &acc, &model);
            expected_avg_exp_search_iterations = acc.get_stat();
        }
        else {
            ExpectedIterationsAndShiftsAccumulator acc(data_capacity);
            build_node_implicit(values, num_keys, data_capacity, &acc, &model);
            expected_avg_exp_search_iterations = acc.get_expected_num_search_iterations();
            expected_avg_shifts = acc.get_expected_num_shifts();
        }
        cost = kExpSearchIterationsWeight * expected_avg_exp_search_iterations +
               kShiftsWeight * expected_avg_shifts * expected_insert_frac;

        if (stats) {
            stats->num_search_iterations = expected_avg_exp_search_iterations;
            stats->num_shifts = expected_avg_shifts;
        }
        return cost;
    }

    // Helper function for compute_expected_cost
    // Implicitly build the data node in order to collect the stats
    static void build_node_implicit(const Value* values, int num_keys, int data_capacity, StatAccumulator* acc,
                                    const LinearModel<double>* model) {
        int last_position = -1;
        int keys_remaining = num_keys;
        for (int i = 0; i < num_keys; i++) {
            int predicted_position = std::max(0, std::min(data_capacity - 1, model->predict(values[i].first)));
            int actual_position = std::max<int>(predicted_position, last_position + 1);
            int positions_remaining = data_capacity - actual_position;
            if (positions_remaining < keys_remaining) {
                actual_position = data_capacity - keys_remaining;
                for (int j = i; j < num_keys; j++) {
                    predicted_position = std::max(0, std::min(data_capacity - 1, model->predict(values[j].first)));
                    acc->accumulate(actual_position, predicted_position);
                    actual_position++;
                }
                break;
            }
            acc->accumulate(actual_position, predicted_position);
            last_position = actual_position;
            keys_remaining--;
        }
    }

    // Computes the expected cost of a data node constructed using the keys
    // between left and right in the key/data_slots of an existing node
    // Assumes existing_model is trained on the dense array of keys
    static double compute_expected_cost_from_existing(AlexDataNode* node, int left, int right, double density,
                         double expected_insert_frac, const LinearModel<double>* existing_model = nullptr,
                                                            DataNodeStats* stats = nullptr) {
        LinearModel<double> model;
        int num_actual_keys = 0;
        if (existing_model == nullptr) {
            Iterator it(node, left);
            LinearModelBuilder<double> builder(&model);
            for (int i = 0; it.cur_idx_ < right && !it.is_end(); it++, i++) {
                builder.add(it.key(), i);
                num_actual_keys++;
            }
            builder.build();
        }
        else {
            num_actual_keys = node->num_keys_in_range(left, right);
            model.a_ = existing_model->a_;
            model.b_ = existing_model->b_;
        }

        if (num_actual_keys == 0) {
            return 0;
        }
        int data_capacity = std::max(static_cast<int>(num_actual_keys / density), num_actual_keys + 1);
        model.expand(static_cast<double>(data_capacity) / num_actual_keys);

        // Compute expected stats in order to compute the expected cost
        double cost = 0;
        double expected_avg_exp_search_iterations = 0;
        double expected_avg_shifts = 0;
        if (expected_insert_frac == 0) {
            ExpectedSearchIterationsAccumulator acc;
            build_node_implicit_from_existing(node, left, right, num_actual_keys, data_capacity, &acc, &model);
            expected_avg_exp_search_iterations = acc.get_stat();
        }
        else {
            ExpectedIterationsAndShiftsAccumulator acc(data_capacity);
            build_node_implicit_from_existing(node, left, right, num_actual_keys, data_capacity, &acc, &model);
            expected_avg_exp_search_iterations = acc.get_expected_num_search_iterations();
            expected_avg_shifts = acc.get_expected_num_shifts();
        }
        cost = kExpSearchIterationsWeight * expected_avg_exp_search_iterations +
               kShiftsWeight * expected_avg_shifts * expected_insert_frac;

        if (stats) {
            stats->num_search_iterations = expected_avg_exp_search_iterations;
            stats->num_shifts = expected_avg_shifts;
        }
        return cost;
    }

    // Helper function for compute_expected_cost
    // Implicitly build the data node in order to collect the stats
    static void build_node_implicit_from_existing(AlexDataNode* node, int left, int right, int num_actual_keys,
                               int data_capacity, StatAccumulator* acc, const LinearModel<double>* model) {
        int last_position = -1;
        int keys_remaining = num_actual_keys;
        Iterator it(node, left);
        for (; it.cur_idx_ < right && !it.is_end(); it++) {
            int predicted_position = std::max(0, std::min(data_capacity - 1, model->predict(it.key())));
            int actual_position = std::max<int>(predicted_position, last_position + 1);
            int positions_remaining = data_capacity - actual_position;
            if (positions_remaining < keys_remaining) {
                actual_position = data_capacity - keys_remaining;
                for (; actual_position < data_capacity; actual_position++, it++) {
                    predicted_position = std::max(0, std::min(data_capacity - 1, model->predict(it.key())));
                    acc->accumulate(actual_position, predicted_position);
                }
                break;
            }
            acc->accumulate(actual_position, predicted_position);
            last_position = actual_position;
            keys_remaining--;
        }
    }

    /*** Bulk loading and model building ***/

    // Assumes pretrained_model is trained on dense array of keys
    void bulk_load(const Value values[], int num_keys, const LinearModel<double>* pretrained_model = nullptr) {
        num_keys_ = num_keys;
        data_capacity_ = std::max(static_cast<int>(num_keys / kInitDensity_), num_keys + 1);
        bitmap_size_ = static_cast<size_t>(std::ceil(data_capacity_ / 64.));
        bitmap_ = new (bitmap_alloc_type(allocator_).allocate(bitmap_size_)) uint64_t[bitmap_size_]();  // initialize to all false
        key_slots_ = new (key_alloc_type(allocator_).allocate(data_capacity_)) double[data_capacity_];
        payload_slots_ = new (payload_alloc_type(allocator_).allocate(data_capacity_)) double[data_capacity_];

        if (num_keys == 0) {
            expansion_threshold_ = data_capacity_;
            contraction_threshold_ = 0;
            for (int i = 0; i < data_capacity_; i++) {
                key_slots_[i] = kEndSentinel_;
            }
            return;
        }

        // Build model
        if (pretrained_model != nullptr) {
            this->model_.a_ = pretrained_model->a_;
            this->model_.b_ = pretrained_model->b_;
        }
        else {
            build_model(values, num_keys, &(this->model_));
        }
        this->model_.expand(static_cast<double>(data_capacity_) / num_keys);

        // Model-based inserts
        int last_position = -1;
        int keys_remaining = num_keys;
        for (int i = 0; i < num_keys; i++) {
            int position = this->model_.predict(values[i].first);
            position = std::max<int>(position, last_position + 1);

            int positions_remaining = data_capacity_ - position;
            if (positions_remaining < keys_remaining) {
                // fill the rest of the store contiguously
                int pos = data_capacity_ - keys_remaining;
                for (int j = last_position + 1; j < pos; j++) {
                    key_slots_[j] = values[i].first;
                }
                for (int j = i; j < num_keys; j++) {
                    key_slots_[pos] = values[j].first;
                    payload_slots_[pos] = values[j].second;
                    set_bit(pos);
                    pos++;
                }
                last_position = pos - 1;
                break;
            }
            for (int j = last_position + 1; j < position; j++) {
                key_slots_[j] = values[i].first;
            }
            key_slots_[position] = values[i].first;
            payload_slots_[position] = values[i].second;
            set_bit(position);
            last_position = position;
            keys_remaining--;
        }
        for (int i = last_position + 1; i < data_capacity_; i++) {
            key_slots_[i] = kEndSentinel_;
        }
        expansion_threshold_ = std::min(std::max(data_capacity_ * kMaxDensity_, static_cast<double>(num_keys + 1)),
                                                                                static_cast<double>(data_capacity_));
        contraction_threshold_ = data_capacity_ * kMinDensity_;
        max_key_ = values[num_keys - 1].first;
    }

    // Bulk load using the keys between the left and right positions in
    // key/data_slots of an existing data node
    // keep_left and keep_right are set if the existing node was append-mostly
    // If the linear model and num_actual_keys have been precomputed, we can avoid
    // redundant work
    void bulk_load_from_existing(AlexDataNode* node, int left, int right, bool keep_left = false, bool keep_right = false,
                                 const LinearModel<double>* precomputed_model = nullptr, int precomputed_num_actual_keys = -1) {
        // Build model
        int num_actual_keys = 0;
        if (precomputed_model == nullptr || precomputed_num_actual_keys == -1) {
            Iterator it(node, left);
            LinearModelBuilder<double> builder(&(this->model_));
            for (int i = 0; it.cur_idx_ < right && !it.is_end(); it++, i++) {
              builder.add(it.key(), i);
              num_actual_keys++;
            }
            builder.build();
        } else {
          num_actual_keys = precomputed_num_actual_keys;
          this->model_.a_ = precomputed_model->a_;
          this->model_.b_ = precomputed_model->b_;
        }

        // Initialization
        num_keys_ = num_actual_keys;
        data_capacity_ = std::max(static_cast<int>(num_actual_keys / kMinDensity_), num_actual_keys + 1);
        bitmap_size_ = static_cast<size_t>(std::ceil(data_capacity_ / 64.));
        bitmap_ = new (bitmap_alloc_type(allocator_).allocate(bitmap_size_)) uint64_t[bitmap_size_]();  // initialize to all false
        key_slots_ = new (key_alloc_type(allocator_).allocate(data_capacity_)) double[data_capacity_];
        payload_slots_ = new (payload_alloc_type(allocator_).allocate(data_capacity_)) double[data_capacity_];

        if (num_actual_keys == 0) {
            expansion_threshold_ = data_capacity_;
            contraction_threshold_ = 0;
            for (int i = 0; i < data_capacity_; i++) {
                key_slots_[i] = kEndSentinel_;
            }
            return;
        }

        // Special casing if existing node was append-mostly
        if (keep_left) {
            this->model_.expand((num_actual_keys / kMaxDensity_) / num_keys_);
        }
        else if (keep_right) {
            this->model_.expand((num_actual_keys / kMaxDensity_) / num_keys_);
            this->model_.b_ += (data_capacity_ - (num_actual_keys / kMaxDensity_));
        }
        else {
            this->model_.expand(static_cast<double>(data_capacity_) / num_keys_);
        }

        // Model-based inserts
        int last_position = -1;
        int keys_remaining = num_keys_;
        Iterator it(node, left);
        for (; it.cur_idx_ < right && !it.is_end(); it++) {
            int position = this->model_.predict(it.key());
            position = std::max<int>(position, last_position + 1);

            int positions_remaining = data_capacity_ - position;
            if (positions_remaining < keys_remaining) {
                // fill the rest of the store contiguously
                int pos = data_capacity_ - keys_remaining;
                for (int j = last_position + 1; j < pos; j++) {
                    key_slots_[j] = it.key();
                }
                for (; pos < data_capacity_; pos++, it++) {
                    key_slots_[pos] = it.key();
                    payload_slots_[pos] = it.payload();
                    set_bit(pos);
                }
                last_position = pos - 1;
                break;
            }

            for (int j = last_position + 1; j < position; j++) {
                key_slots_[j] = it.key();
            }
            key_slots_[position] = it.key();
            payload_slots_[position] = it.payload();
            set_bit(position);
            last_position = position;
            keys_remaining--;
        }

        for (int i = last_position + 1; i < data_capacity_; i++) {
            key_slots_[i] = kEndSentinel_;
        }
        max_key_ = node->max_key_;
        expansion_threshold_ = std::min(std::max(data_capacity_ * kMaxDensity_, static_cast<double>(num_keys_ + 1)),
                                                                                static_cast<double>(data_capacity_));
        contraction_threshold_ = data_capacity_ * kMinDensity_;
    }

    static void build_model(const Value* values, int num_keys, LinearModel<double>* model, bool use_sampling = false) {
        LinearModelBuilder<double> builder(model);
        for (int i = 0; i < num_keys; i++) {
            builder.add(values[i].first, i);
        }
        builder.build();
    }

    /*** Lookup ***/
    // Predicts the position of a key using the model
    inline int predict_position(const double& key) const {
        int position = this->model_.predict(key);
        position = std::max<int>(std::min<int>(position, data_capacity_ - 1), 0);
        return position;
    }
    // Searches for the last non-gap position equal to key. If no positions equal to key, returns -1
    int find_key(const double& key) {
        num_lookups_++;
        int predicted_pos = predict_position(key);

        // The last key slot with a certain value is guaranteed to be a real key (instead of a gap)
        int pos = exponential_search_upper_bound(predicted_pos, key) - 1;
        if (pos < 0 || !key_equal(key_slots_[pos], key)) {
            return -1;
        }
        else {
            return pos;
        }
    }
    // Searches for the first non-gap position no less than key
    // Returns position in range [0, data_capacity]
    // Compare with lower_bound()
    int find_lower(const double& key) {
        num_lookups_++;
        int predicted_pos = predict_position(key);

        int pos = exponential_search_lower_bound(predicted_pos, key);
        return get_next_filled_position(pos, false);
    }
    // Searches for the first non-gap position greater than key
    // Returns position in range [0, data_capacity]
    // Compare with upper_bound()
    int find_upper(const double& key) {
        num_lookups_++;
        int predicted_pos = predict_position(key);

        int pos = exponential_search_upper_bound(predicted_pos, key);
        return get_next_filled_position(pos, false);
    }
    // Finds position to insert a key.
    // First returned value takes prediction into account.
    // Second returned value is first valid position (i.e., upper_bound of key).
    // If there are duplicate keys, the insert position will be to the right of all existing keys of the same value.
    std::pair<int, int> find_insert_position(const double& key) {
        int predicted_pos = predict_position(key);  // first use model to get prediction

        // insert to the right of duplicate keys
        int pos = exponential_search_upper_bound(predicted_pos, key);
        if (predicted_pos <= pos || check_exists(pos)) {
            return {pos, pos};
        }
        else {
            // Place inserted key as close as possible to the predicted position while
            // maintaining correctness
            return {std::min(predicted_pos, get_next_filled_position(pos, true) - 1), pos};
        }
    }
    // Starting from a position, return the first position that is not a gap
    // If no more filled positions, will return data_capacity
    // If exclusive is true, output is at least (pos + 1)
    // If exclusive is false, output can be pos itself
    int get_next_filled_position(int pos, bool exclusive) const {
        if (exclusive) {
            pos++;
            if (pos == data_capacity_) {
                return data_capacity_;
            }
        }
        int curBitmapIdx = pos >> 6;
        uint64_t curBitmapData = bitmap_[curBitmapIdx];

        // Zero out extra bits
        int bit_pos = pos - (curBitmapIdx << 6);
        curBitmapData &= ~((1ULL << (bit_pos)) - 1);

        while (curBitmapData == 0) {
            curBitmapIdx++;
            if (curBitmapIdx >= bitmap_size_) {
                return data_capacity_;
            }
            curBitmapData = bitmap_[curBitmapIdx];
        }
        uint64_t bit = extract_rightmost_one(curBitmapData);
        return get_offset(curBitmapIdx, bit);
    }
    // Searches for the first position greater than key
    // This could be the position for a gap (i.e., its bit in the bitmap is 0)
    // Returns position in range [0, data_capacity]
    // Compare with find_upper()
    int upper_bound(const double& key) {
        num_lookups_++;
        int position = predict_position(key);
        return exponential_search_upper_bound(position, key);
    }
    // Searches for the first position greater than key, starting from position m
    // Returns position in range [0, data_capacity]
    inline int exponential_search_upper_bound(int m, const double& key) {
        // Continue doubling the bound until it contains the upper bound. Then use
        // binary search.
        int bound = 1;
        int l, r;  // will do binary search in range [l, r)
        if (key_greater(key_slots_[m], key)) {
            int size = m;
            while (bound < size && key_greater(key_slots_[m - bound], key)) {
                bound <<= 1;
                num_exp_search_iterations_++;
            }
            l = m - std::min<int>(bound, size);
            r = m - (bound >> 1);
        }
        else {
            int size = data_capacity_ - m;
            while (bound < size && key_lessequal(key_slots_[m + bound], key)) {
                bound <<= 1;
                num_exp_search_iterations_++;
            }
            l = m + (bound >> 1);
            r = m + std::min<int>(bound, size);
        }
        return binary_search_upper_bound(l, r, key);
    }
    // Searches for the first position greater than key in range [l, r)
    // https://stackoverflow.com/questions/6443569/implementation-of-c-lower-bound
    // Returns position in range [l, r]
    inline int binary_search_upper_bound(int l, int r, const double& key) const {
        while (l < r) {
            int mid = l + ((r - l) >> 1);
            if (key_lessequal(key_slots_[mid], key)) {
                l = mid + 1;
            }
            else {
                r = mid;
            }
        }
        return l;
    }
    // Searches for the first position no less than key
    // This could be the position for a gap (i.e., its bit in the bitmap is 0)
    // Returns position in range [0, data_capacity]
    // Compare with find_lower()
    int lower_bound(const double& key) {
        num_lookups_++;
        int position = predict_position(key);
        return exponential_search_lower_bound(position, key);
    }
    // Searches for the first position no less than key, starting from position m
    // Returns position in range [0, data_capacity]
    inline int exponential_search_lower_bound(int m, const double& key) {
        // Continue doubling the bound until it contains the lower bound. Then use binary search.
        int bound = 1;
        int l, r;  // will do binary search in range [l, r)
        if (key_greaterequal(key_slots_[m], key)) {
            int size = m;
            while (bound < size && key_greaterequal(key_slots_[m - bound], key)) {
                bound <<= 1;
                num_exp_search_iterations_++;
            }
            l = m - std::min<int>(bound, size);
            r = m - (bound >> 1);
        }
        else {
            int size = data_capacity_ - m;
            while (bound < size && key_less(key_slots_[m + bound], key)) {
                bound <<= 1;
                num_exp_search_iterations_++;
            }
            l = m + (bound >> 1);
            r = m + std::min<int>(bound, size);
        }
        return binary_search_lower_bound(l, r, key);
    }
    // Searches for the first position no less than key in range [l, r)
    // https://stackoverflow.com/questions/6443569/implementation-of-c-lower-bound
    // Returns position in range [l, r]
    inline int binary_search_lower_bound(int l, int r, const double& key) const {
        while (l < r) {
            int mid = l + ((r - l) >> 1);
            if (key_greaterequal(key_slots_[mid], key)) {
                r = mid;
            }
            else {
                l = mid + 1;
            }
        }
        return l;
    }

    /*** Inserts and resizes ***/
    // Whether empirical cost deviates significantly from expected cost
    // Also returns false if empirical cost is sufficiently low and is not worth splitting
    inline bool significant_cost_deviation() const {
        double emp_cost = empirical_cost();
        return emp_cost > kNodeLookupsWeight && emp_cost > 1.5 * this->cost_;
    }
    // Returns true if cost is catastrophically high and we want to force a split
    // The heuristic for this is if the number of shifts per insert (expected or empirical) is over 100
    inline bool catastrophic_cost() const {
        return shifts_per_insert() > 100 || expected_avg_shifts_ > 100;
    }

    // First value in returned pair is fail flag:
    // 0 if successful insert (possibly with automatic expansion).
    // 1 if no insert because of significant cost deviation.
    // 2 if no insert because of "catastrophic" cost.
    // 3 if no insert because node is at max capacity.
    // -1 if key already exists and duplicates not allowed.

    // Second value in returned pair is position of inserted key, or of the already-existing key.
    // -1 if no insertion.
    std::pair<int, int> insert(const double& key, const double& payload) {
        // Periodically check for catastrophe
        if (num_inserts_ % 64 == 0 && catastrophic_cost()) {
            return {2, -1};
        }
        // Check if node is full (based on expansion_threshold)
        if (num_keys_ >= expansion_threshold_) {
            if (significant_cost_deviation()) {
                return {1, -1};
            }
            if (catastrophic_cost()) {
                return {2, -1};
            }
            if (num_keys_ > max_slots_ * kMinDensity_) {
                return {3, -1};
            }
            // Expand
            bool keep_left = is_append_mostly_right();
            bool keep_right = is_append_mostly_left();
            resize(kMinDensity_, false, keep_left, keep_right);
            num_resizes_++;
        }
        // Insert
        std::pair<int, int> positions = find_insert_position(key);
        int upper_bound_pos = positions.second;
        int insertion_position = positions.first;
        if (insertion_position < data_capacity_ && !check_exists(insertion_position)) {
            insert_element_at(key, payload, insertion_position);
        }
        else {
            insertion_position = insert_using_shifts(key, payload, insertion_position);
        }
        // Update stats
        num_keys_++;
        num_inserts_++;
        if (key > max_key_) {
            max_key_ = key;
            num_right_out_of_bounds_inserts_++;
        }
        if (key < min_key_) {
            min_key_ = key;
            num_left_out_of_bounds_inserts_++;
        }
        return {0, insertion_position};
    }

    // Resize the data node to the target density
    void resize(double target_density, bool force_retrain = false, bool keep_left = false, bool keep_right = false) {
        if (num_keys_ == 0) {
            return;
        }
        int new_data_capacity = std::max(static_cast<int>(num_keys_ / target_density), num_keys_ + 1);
        auto new_bitmap_size = static_cast<size_t>(std::ceil(new_data_capacity / 64.));
        auto new_bitmap = new (bitmap_alloc_type(allocator_).allocate(new_bitmap_size)) uint64_t[new_bitmap_size]();
        double* new_key_slots = new (key_alloc_type(allocator_).allocate(new_data_capacity)) double[new_data_capacity];
        double* new_payload_slots = new (payload_alloc_type(allocator_).allocate(new_data_capacity)) double[new_data_capacity];

        // Retrain model if the number of keys is sufficiently small (under 50)
        if (num_keys_ < 50 || force_retrain) {
            Iterator it(this, 0);
            LinearModelBuilder<double> builder(&(this->model_));
            for (int i = 0; it.cur_idx_ < data_capacity_ && !it.is_end(); it++, i++) {
                builder.add(it.key(), i);
            }
            builder.build();
            if (keep_left) {
                this->model_.expand(static_cast<double>(data_capacity_) / num_keys_);
            }
            else if (keep_right) {
                this->model_.expand(static_cast<double>(data_capacity_) / num_keys_);
                this->model_.b_ += (new_data_capacity - data_capacity_);
            }
            else {
                this->model_.expand(static_cast<double>(new_data_capacity) / num_keys_);
            }
        }
        else {
            if (keep_right) {
                this->model_.b_ += (new_data_capacity - data_capacity_);
            }
            else if (!keep_left) {
                this->model_.expand(static_cast<double>(new_data_capacity) / data_capacity_);
            }
        }
        int last_position = -1;
        int keys_remaining = num_keys_;
        Iterator it(this, 0);
        for (; it.cur_idx_ < data_capacity_ && !it.is_end(); it++) {
            int position = this->model_.predict(it.key());
            position = std::max<int>(position, last_position + 1);

            int positions_remaining = new_data_capacity - position;
            if (positions_remaining < keys_remaining) {
                // fill the rest of the store contiguously
                int pos = new_data_capacity - keys_remaining;
                for (int j = last_position + 1; j < pos; j++) {
                    new_key_slots[j] = it.key();
                }
                for (; pos < new_data_capacity; pos++, it++) {
                    new_key_slots[pos] = it.key();
                    new_payload_slots[pos] = it.payload();
                    set_bit(new_bitmap, pos);
                }
                last_position = pos - 1;
                break;
            }

            for (int j = last_position + 1; j < position; j++) {
                new_key_slots[j] = it.key();
            }
            new_key_slots[position] = it.key();
            new_payload_slots[position] = it.payload();
            set_bit(new_bitmap, position);
            last_position = position;
            keys_remaining--;
        }
        for (int i = last_position + 1; i < new_data_capacity; i++) {
            new_key_slots[i] = kEndSentinel_;
        }

        key_alloc_type(allocator_).deallocate(key_slots_, data_capacity_);
        payload_alloc_type(allocator_).deallocate(payload_slots_, data_capacity_);
        bitmap_alloc_type(allocator_).deallocate(bitmap_, bitmap_size_);

        data_capacity_ = new_data_capacity;
        bitmap_size_ = new_bitmap_size;
        key_slots_ = new_key_slots;
        payload_slots_ = new_payload_slots;
        bitmap_ = new_bitmap;

        expansion_threshold_ = std::min(std::max(data_capacity_ * kMaxDensity_, static_cast<double>(num_keys_ + 1)),
                                                                                static_cast<double>(data_capacity_));
        contraction_threshold_ = data_capacity_ * kMinDensity_;
    }

    inline bool is_append_mostly_right() const {
        return static_cast<double>(num_right_out_of_bounds_inserts_) / num_inserts_ > kAppendMostlyThreshold;
    }

    inline bool is_append_mostly_left() const {
        return static_cast<double>(num_left_out_of_bounds_inserts_) / num_inserts_ > kAppendMostlyThreshold;
    }

    // Insert key into pos. The caller must guarantee that pos is a gap.
    void insert_element_at(const double& key, double payload, int pos) {
        key_slots_[pos] = key;
        payload_slots_[pos] = payload;
        set_bit(pos);

        // Overwrite preceding gaps until we reach the previous element
        pos--;
        while (pos >= 0 && !check_exists(pos)) {
            key_slots_[pos] = key;
            pos--;
        }
    }

    // Insert key into pos, shifting as necessary in the range [left, right)
    // Returns the actual position of insertion
    int insert_using_shifts(const double& key, double payload, int pos) {
        // Find the closest gap
        int gap_pos = closest_gap(pos);
        set_bit(gap_pos);
        if (gap_pos >= pos) {
            for (int i = gap_pos; i > pos; i--) {
                key_slots_[i] = key_slots_[i - 1];
                payload_slots_[i] = payload_slots_[i - 1];
            }
            insert_element_at(key, payload, pos);
            num_shifts_ += gap_pos - pos;
            return pos;
        }
        else {
            for (int i = gap_pos; i < pos - 1; i++) {
                key_slots_[i] = key_slots_[i + 1];
                payload_slots_[i] = payload_slots_[i + 1];
            }
            insert_element_at(key, payload, pos - 1);
            num_shifts_ += pos - gap_pos - 1;
            return pos - 1;
        }
    }

    // Does not return pos if pos is a gap
    int closest_gap(int pos) const {
        int max_left_offset = pos;
        int max_right_offset = data_capacity_ - pos - 1;
        int max_bidirectional_offset = std::min<int>(max_left_offset, max_right_offset);
        int distance = 1;
        while (distance <= max_bidirectional_offset) {
            if (!check_exists(pos - distance)) {
                return pos - distance;
            }
            if (!check_exists(pos + distance)) {
                return pos + distance;
            }
            distance++;
        }
        if (max_left_offset > max_right_offset) {
            for (int i = pos - distance; i >= 0; i--) {
                if (!check_exists(i)) return i;
            }
        }
        else {
            for (int i = pos + distance; i < data_capacity_; i++) {
                if (!check_exists(i)) return i;
            }
        }
        return -1;
    }

    /*** Deletes ***/
    // Erase the left-most key with the input value
    // Returns the number of keys erased (0 or 1)
    int erase_one(const double& key) {
        int pos = find_lower(key);
        if (pos == data_capacity_ || !key_equal(key_slots_[pos], key))
            return 0;
        // Erase key at pos
        erase_one_at(pos);
        return 1;
    }
    // Erase the key at the given position
    void erase_one_at(int pos) {
        double next_key;
        if (pos == data_capacity_ - 1) {
            next_key = kEndSentinel_;
        }
        else {
            next_key = key_slots_[pos + 1];
        }
        key_slots_[pos] = next_key;
        unset_bit(pos);
        pos--;

        // Erase preceding gaps until we reach an existing key
        while (pos >= 0 && !check_exists(pos)) {
            key_slots_[pos] = next_key;
            pos--;
        }
        num_keys_--;
        if (num_keys_ < contraction_threshold_) {
            resize(kMaxDensity_);  // contract
            num_resizes_++;
        }
    }
    // Erase all keys with the input value
    // Returns the number of keys erased (there may be multiple keys with the same value)
    int erase(const double& key) {
        int pos = upper_bound(key);
        if (pos == 0 || !key_equal(key_slots_[pos - 1], key)) return 0;

        // Erase preceding positions until we reach a key with smaller value
        int num_erased = 0;
        double next_key;
        if (pos == data_capacity_) {
            next_key = kEndSentinel_;
        }
        else {
            next_key = key_slots_[pos];
        }
        pos--;
        while (pos >= 0 && key_equal(key_slots_[pos], key)) {
            key_slots_[pos] = next_key;
            num_erased += check_exists(pos);
            unset_bit(pos);
            pos--;
        }
        num_keys_ -= num_erased;
        if (num_keys_ < contraction_threshold_) {
            resize(kMaxDensity_);  // contract
            num_resizes_++;
        }
        return num_erased;
    }
    // Erase keys with value between start key (inclusive) and end key.
    // Returns the number of keys erased.
    int erase_range(double start_key, double end_key, bool end_key_inclusive = false) {
        int pos;
        if (end_key_inclusive) {
            pos = upper_bound(end_key);
        }
        else {
            pos = lower_bound(end_key);
        }
        if (pos == 0) return 0;

        // Erase preceding positions until key value is below the start key
        int num_erased = 0;
        double next_key;
        if (pos == data_capacity_) {
            next_key = kEndSentinel_;
        }
        else {
            next_key = key_slots_[pos];
        }
        pos--;
        while (pos >= 0 && key_greaterequal(key_slots_[pos], start_key)) {
            key_slots_[pos] = next_key;
            num_erased += check_exists(pos);
            unset_bit(pos);
            pos--;
        }
        num_keys_ -= num_erased;
        if (num_keys_ < contraction_threshold_) {
            resize(kMaxDensity_);  // contract
            num_resizes_++;
        }
        return num_erased;
    }

    /*** Stats ***/
    // Total size of node metadata
    long long node_size() const override { return sizeof(AlexDataNode); }

    // Total size in bytes of key/payload/data_slots and bitmap
    long long data_size() const {
        long long data_size = data_capacity_ * sizeof(double); // key slots
        data_size += data_capacity_ * sizeof(double); // payload slots
        data_size += bitmap_size_ * sizeof(uint64_t);
        return data_size;
    }
};
}