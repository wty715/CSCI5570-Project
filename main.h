#include "alex_base.h"
#include "alex_fanout_tree.h"
#include "node.h"
#include <stack>

namespace alex {
typedef std::pair<double, double> Value;
class Alex {
    public:
    // value type define
    typedef std::pair<double, double> Value;

    // Forward declaration for iterators
    class Iterator;
    class ConstIterator;
    class ReverseIterator;
    class ConstReverseIterator;
    class NodeIterator;  // Iterates through all nodes with pre-order traversal

    AlexNode* root_node_ = nullptr;
    AlexModelNode* superroot_ = nullptr;  // phantom node that is the root's parent

    // When bulk loading, Alex can use provided knowledge of the expected fraction of operations that will be inserts
    // For simplicity, operations are either point lookups ("reads") or inserts ("writes)
    // i.e., 0 means we expect a read-only workload, 1 means write-only
    const double expected_insert_frac = 0.5;
    // Maximum node size, in bytes. By default, 16MB.
    const int max_node_size = 1 << 24;

    /* Setting max node size automatically changes these parameters */
    struct DerivedParams {
      // The defaults here assume the default max node size of 16MB
      int max_fanout = 1 << 21;  // assumes 8-byte pointers
      int max_data_node_slots = (1 << 24) / sizeof(Value);
    } derived_params_;
    
    // Counters, useful for benchmarking and profiling
    struct Stats {
        int num_keys = 0;
        int num_model_nodes = 0;  // num of model nodes
        int num_data_nodes = 0;   // num of data nodes
        int num_expand_and_scales = 0;
        int num_expand_and_retrains = 0;
        int num_downward_splits = 0;
        int num_sideways_splits = 0;
        int num_model_node_expansions = 0;
        int num_model_node_splits = 0;
        long long num_downward_split_keys = 0;
        long long num_sideways_split_keys = 0;
        long long num_model_node_expansion_pointers = 0;
        long long num_model_node_split_pointers = 0;
        mutable long long num_node_lookups = 0;
        mutable long long num_lookups = 0;
        long long num_inserts = 0;
        double splitting_time = 0;
        double cost_computation_time = 0;
    } stats_;

    struct ExperimentalParams {
        // Fanout selection method used during bulk loading: 0 means use bottom-up
        // fanout tree, 1 means top-down
        int fanout_selection_method = 0;
        // Policy when a data node experiences significant cost deviation.
        // 0 means always split node in 2
        // 1 means decide between no splitting or splitting in 2
        // 2 means use a full fanout tree to decide the splitting strategy
        int splitting_policy_method = 1;
        // Splitting upwards means that a split can propagate all the way up to the
        // root, like a B+ tree
        // Splitting upwards can result in a better RMI, but has much more overhead
        // than splitting sideways
        bool allow_splitting_upwards = false;
    };
    ExperimentalParams experimental_params_;

    private:
    // Statistics related to the key domain.
    // The index can hold keys outside the domain, but lookups/inserts on those keys will be inefficient.
    // If enough keys fall outside the key domain, then we expand the key domain.
    struct InternalStats {
        double key_domain_min_ = std::numeric_limits<double>::max();
        double key_domain_max_ = std::numeric_limits<double>::lowest();
        int num_keys_above_key_domain = 0;
        int num_keys_below_key_domain = 0;
        int num_keys_at_last_right_domain_resize = 0;
        int num_keys_at_last_left_domain_resize = 0;
    };
    InternalStats istats_;

    // Save the traversal path down the RMI by having a linked list of these structs.
    struct TraversalNode {
        AlexModelNode* node = nullptr;
        int bucketID = -1;
    };

    // Used when finding the best way to propagate up the RMI when splitting upwards.
    // Cost is in terms of additional model size created through splitting upwards, measured in units of pointers.
    // One instance of this struct is created for each node on the traversal path.
    // User should take into account the cost of metadata for new model nodes (base_cost).
    struct SplitDecisionCosts {
        static constexpr double base_cost = static_cast<double>(sizeof(AlexModelNode)) / sizeof(void*);
        // Additional cost due to this node if propagation stops at this node.
        // Equal to 0 if redundant slot exists, otherwise number of new pointers due to node expansion.
        double stop_cost = 0;
        // Additional cost due to this node if propagation continues past this node.
        // Equal to number of new pointers due to node splitting, plus size of metadata of new model node.
        double split_cost = 0;
    };

    // At least this many keys must be outside the domain before a domain expansion is triggered.
    static const int kMinOutOfDomainKeys = 5;
    // After this many keys are outside the domain, a domain expansion must be triggered.
    static const int kMaxOutOfDomainKeys = 1000;
    // When the number of max out-of-domain (OOD) keys is between the min and max
    // expand the domain if the number of OOD keys is greater than the expected
    // number of OOD due to randomness by greater than the tolereance factor.
    static const int kOutOfDomainToleranceFactor = 2;

    AlexCompare key_less_ = AlexCompare();
    std::allocator<std::pair<double, double>> allocator_ = std::allocator<std::pair<double, double>>();

    // Deep copy of tree starting at given node
    AlexNode* copy_tree_recursive(const AlexNode* node) {
        if (!node) return nullptr;
        if (node->is_leaf_) {
            return new (AlexDataNode::alloc_type(allocator_).allocate(1)) AlexDataNode(*static_cast<const AlexDataNode*>(node));
        }
        else {
            auto node_copy = new (AlexModelNode::alloc_type(allocator_).allocate(1))
                                  AlexModelNode(*static_cast<const AlexModelNode*>(node));
            int cur = 0;
            while (cur < node_copy->num_children_) {
                AlexNode* child_node = node_copy->children_[cur];
                AlexNode* child_node_copy = copy_tree_recursive(child_node);
                int repeats = 1 << child_node_copy->duplication_factor_;
                for (int i = cur; i < cur + repeats; i++) {
                    node_copy->children_[i] = child_node_copy;
                }
                cur += repeats;
            }
            return node_copy;
        }
    }
    
    /*** Iterators ***/
    public:
    class Iterator {
        public:
        AlexDataNode* cur_leaf_ = nullptr;  // current data node
        int cur_idx_ = 0;                   // current position in key/data_slots of data node
        int cur_bitmap_idx_ = 0;            // current position in bitmap
        uint64_t cur_bitmap_data_ = 0;      // caches the relevant data in the current bitmap position

        // Constructors
        Iterator() {}
        Iterator(AlexDataNode* leaf, int idx) : cur_leaf_(leaf), cur_idx_(idx) {
            initialize();
        }
        Iterator(const Iterator& other) : cur_leaf_(other.cur_leaf_), cur_idx_(other.cur_idx_),
                                          cur_bitmap_idx_(other.cur_bitmap_idx_), cur_bitmap_data_(other.cur_bitmap_data_) {}
        Iterator(const ReverseIterator& other) : cur_leaf_(other.cur_leaf_), cur_idx_(other.cur_idx_) {
            initialize();
        }
        // Operators
        Iterator& operator=(const Iterator& other) {
            if (this != &other) {
                cur_idx_ = other.cur_idx_;
                cur_leaf_ = other.cur_leaf_;
                cur_bitmap_idx_ = other.cur_bitmap_idx_;
                cur_bitmap_data_ = other.cur_bitmap_data_;
            }
            return *this;
        }
        Iterator& operator++() {
            advance();
            return *this;
        }
        Iterator operator++(int) {
            Iterator tmp = *this;
            advance();
            return tmp;
        }
        // Does not return a reference because keys and payloads are stored separately.
        // If possible, use key() and payload() instead.
        Value operator*() const {
            return std::make_pair(cur_leaf_->key_slots_[cur_idx_], cur_leaf_->payload_slots_[cur_idx_]);
        }
        const double& key() const { return cur_leaf_->get_key(cur_idx_); }
        double& payload() const { return cur_leaf_->get_payload(cur_idx_); }
        bool is_end() const { return cur_leaf_ == nullptr; }
        bool operator==(const Iterator& rhs) const {
            return cur_idx_ == rhs.cur_idx_ && cur_leaf_ == rhs.cur_leaf_;
        }
        bool operator!=(const Iterator& rhs) const { return !(*this == rhs); };

        private:
        void initialize() {
            if (!cur_leaf_) return;
            assert(cur_idx_ >= 0);
            if (cur_idx_ >= cur_leaf_->data_capacity_) {
                cur_leaf_ = cur_leaf_->next_leaf_;
                cur_idx_ = 0;
                if (!cur_leaf_) return;
            }
            cur_bitmap_idx_ = cur_idx_ >> 6;
            cur_bitmap_data_ = cur_leaf_->bitmap_[cur_bitmap_idx_];
            // Zero out extra bits
            int bit_pos = cur_idx_ - (cur_bitmap_idx_ << 6);
            cur_bitmap_data_ &= ~((1ULL << bit_pos) - 1);
            (*this)++;
        }
        inline void advance() {
            while (cur_bitmap_data_ == 0) {
                cur_bitmap_idx_++;
                if (cur_bitmap_idx_ >= cur_leaf_->bitmap_size_) {
                    cur_leaf_ = cur_leaf_->next_leaf_;
                    cur_idx_ = 0;
                    if (cur_leaf_ == nullptr) return;
                    cur_bitmap_idx_ = 0;
                }
                cur_bitmap_data_ = cur_leaf_->bitmap_[cur_bitmap_idx_];
            }
            uint64_t bit = extract_rightmost_one(cur_bitmap_data_);
            cur_idx_ = get_offset(cur_bitmap_idx_, bit);
            cur_bitmap_data_ = remove_rightmost_one(cur_bitmap_data_);
        }
    };

    class ConstIterator {
        public:
        const AlexDataNode* cur_leaf_ = nullptr;    // current data node
        int cur_idx_ = 0;                           // current position in key/data_slots of data node
        int cur_bitmap_idx_ = 0;                    // current position in bitmap
        uint64_t cur_bitmap_data_ = 0;              // caches the relevant data in the current bitmap position

        ConstIterator() {}
        ConstIterator(const AlexDataNode* leaf, int idx) : cur_leaf_(leaf), cur_idx_(idx) {
            initialize();
        }
        ConstIterator(const Iterator& other) : cur_leaf_(other.cur_leaf_), cur_idx_(other.cur_idx_),
                                               cur_bitmap_idx_(other.cur_bitmap_idx_), cur_bitmap_data_(other.cur_bitmap_data_) {}
        ConstIterator(const ConstIterator& other) : cur_leaf_(other.cur_leaf_), cur_idx_(other.cur_idx_),
                                                    cur_bitmap_idx_(other.cur_bitmap_idx_),
                                                    cur_bitmap_data_(other.cur_bitmap_data_) {}
        ConstIterator(const ReverseIterator& other) : cur_leaf_(other.cur_leaf_), cur_idx_(other.cur_idx_) {
            initialize();
        }
        ConstIterator(const ConstReverseIterator& other) : cur_leaf_(other.cur_leaf_), cur_idx_(other.cur_idx_) {
            initialize();
        }

        ConstIterator& operator=(const ConstIterator& other) {
            if (this != &other) {
                cur_idx_ = other.cur_idx_;
                cur_leaf_ = other.cur_leaf_;
                cur_bitmap_idx_ = other.cur_bitmap_idx_;
                cur_bitmap_data_ = other.cur_bitmap_data_;
            }
            return *this;
        }
        ConstIterator& operator++() {
            advance();
            return *this;
        }
        ConstIterator operator++(int) {
            ConstIterator tmp = *this;
            advance();
            return tmp;
        }
        // Does not return a reference because keys and payloads are stored separately.
        // If possible, use key() and payload() instead.
        Value operator*() const {
            return std::make_pair(cur_leaf_->key_slots_[cur_idx_], cur_leaf_->payload_slots_[cur_idx_]);
        }
        const double& key() const { return cur_leaf_->get_key(cur_idx_); }
        const double& payload() const { return cur_leaf_->get_payload(cur_idx_); }
        bool is_end() const { return cur_leaf_ == nullptr; }
        bool operator==(const ConstIterator& rhs) const {
            return cur_idx_ == rhs.cur_idx_ && cur_leaf_ == rhs.cur_leaf_;
        }
        bool operator!=(const ConstIterator& rhs) const { return !(*this == rhs); };

        private:
        void initialize() {
            if (!cur_leaf_) return;
            assert(cur_idx_ >= 0);
            if (cur_idx_ >= cur_leaf_->data_capacity_) {
                cur_leaf_ = cur_leaf_->next_leaf_;
                cur_idx_ = 0;
                if (!cur_leaf_) return;
            }
            cur_bitmap_idx_ = cur_idx_ >> 6;
            cur_bitmap_data_ = cur_leaf_->bitmap_[cur_bitmap_idx_];
            // Zero out extra bits
            int bit_pos = cur_idx_ - (cur_bitmap_idx_ << 6);
            cur_bitmap_data_ &= ~((1ULL << bit_pos) - 1);
            (*this)++;
        }
        inline void advance() {
            while (cur_bitmap_data_ == 0) {
                cur_bitmap_idx_++;
                if (cur_bitmap_idx_ >= cur_leaf_->bitmap_size_) {
                    cur_leaf_ = cur_leaf_->next_leaf_;
                    cur_idx_ = 0;
                    if (cur_leaf_ == nullptr) {
                      return;
                    }
                    cur_bitmap_idx_ = 0;
                }
                cur_bitmap_data_ = cur_leaf_->bitmap_[cur_bitmap_idx_];
            }
            uint64_t bit = extract_rightmost_one(cur_bitmap_data_);
            cur_idx_ = get_offset(cur_bitmap_idx_, bit);
            cur_bitmap_data_ = remove_rightmost_one(cur_bitmap_data_);
        }
    };

    class ReverseIterator {
        public:
        AlexDataNode* cur_leaf_ = nullptr;    // current data node
        int cur_idx_ = 0;                     // current position in key/data_slots of data node
        int cur_bitmap_idx_ = 0;              // current position in bitmap
        uint64_t cur_bitmap_data_ = 0;        // caches the relevant data in the current bitmap position

        ReverseIterator() {}
        ReverseIterator(AlexDataNode* leaf, int idx) : cur_leaf_(leaf), cur_idx_(idx) {
            initialize();
        }
        ReverseIterator(const ReverseIterator& other) : cur_leaf_(other.cur_leaf_), cur_idx_(other.cur_idx_),
                                                        cur_bitmap_idx_(other.cur_bitmap_idx_),
                                                        cur_bitmap_data_(other.cur_bitmap_data_) {}
        ReverseIterator(const Iterator& other) : cur_leaf_(other.cur_leaf_), cur_idx_(other.cur_idx_) {
            initialize();
        }

        ReverseIterator& operator=(const ReverseIterator& other) {
            if (this != &other) {
                cur_idx_ = other.cur_idx_;
                cur_leaf_ = other.cur_leaf_;
                cur_bitmap_idx_ = other.cur_bitmap_idx_;
                cur_bitmap_data_ = other.cur_bitmap_data_;
            }
            return *this;
        }
        ReverseIterator& operator++() {
            advance();
            return *this;
        }
        ReverseIterator operator++(int) {
            ReverseIterator tmp = *this;
            advance();
            return tmp;
        }
        // Does not return a reference because keys and payloads are stored separately.
        // If possible, use key() and payload() instead.
        Value operator*() const {
            return std::make_pair(cur_leaf_->key_slots_[cur_idx_], cur_leaf_->payload_slots_[cur_idx_]);
        }
        const double& key() const { return cur_leaf_->get_key(cur_idx_); }
        double& payload() const { return cur_leaf_->get_payload(cur_idx_); }
        bool is_end() const { return cur_leaf_ == nullptr; }
        bool operator==(const ReverseIterator& rhs) const {
            return cur_idx_ == rhs.cur_idx_ && cur_leaf_ == rhs.cur_leaf_;
        }
        bool operator!=(const ReverseIterator& rhs) const {
            return !(*this == rhs);
        };

        private:
        void initialize() {
            if (!cur_leaf_) return;
            assert(cur_idx_ >= 0);
            if (cur_idx_ >= cur_leaf_->data_capacity_) {
                cur_leaf_ = cur_leaf_->next_leaf_;
                cur_idx_ = 0;
                if (!cur_leaf_) return;
            }
            cur_bitmap_idx_ = cur_idx_ >> 6;
            cur_bitmap_data_ = cur_leaf_->bitmap_[cur_bitmap_idx_];
            // Zero out extra bits
            int bit_pos = cur_idx_ - (cur_bitmap_idx_ << 6);
            cur_bitmap_data_ &= (1ULL << bit_pos) | ((1ULL << bit_pos) - 1);
            advance();
        }

        inline void advance() {
            while (cur_bitmap_data_ == 0) {
                cur_bitmap_idx_--;
                if (cur_bitmap_idx_ < 0) {
                    cur_leaf_ = cur_leaf_->prev_leaf_;
                    if (cur_leaf_ == nullptr) {
                        cur_idx_ = 0;
                        return;
                    }
                    cur_idx_ = cur_leaf_->data_capacity_ - 1;
                    cur_bitmap_idx_ = cur_leaf_->bitmap_size_ - 1;
                }
                cur_bitmap_data_ = cur_leaf_->bitmap_[cur_bitmap_idx_];
            }
            assert(cpu_supports_bmi());
            int bit_pos = static_cast<int>(63 - _lzcnt_u64(cur_bitmap_data_));
            cur_idx_ = (cur_bitmap_idx_ << 6) + bit_pos;
            cur_bitmap_data_ &= ~(1ULL << bit_pos);
        }
    };

    class ConstReverseIterator {
        public:
        const AlexDataNode* cur_leaf_ = nullptr;  // current data node
        int cur_idx_ = 0;                         // current position in key/data_slots of data node
        int cur_bitmap_idx_ = 0;                  // current position in bitmap
        uint64_t cur_bitmap_data_ = 0;            // caches the relevant data in the current bitmap position

        ConstReverseIterator() {}
        ConstReverseIterator(const AlexDataNode* leaf, int idx) : cur_leaf_(leaf), cur_idx_(idx) {
            initialize();
        }
        ConstReverseIterator(const ConstReverseIterator& other) : cur_leaf_(other.cur_leaf_), cur_idx_(other.cur_idx_),
                                                                  cur_bitmap_idx_(other.cur_bitmap_idx_),
                                                                  cur_bitmap_data_(other.cur_bitmap_data_) {}
        ConstReverseIterator(const ReverseIterator& other) : cur_leaf_(other.cur_leaf_), cur_idx_(other.cur_idx_),
                                                             cur_bitmap_idx_(other.cur_bitmap_idx_),
                                                             cur_bitmap_data_(other.cur_bitmap_data_) {}
        ConstReverseIterator(const Iterator& other) : cur_leaf_(other.cur_leaf_), cur_idx_(other.cur_idx_) {
            initialize();
        }

        ConstReverseIterator(const ConstIterator& other) : cur_leaf_(other.cur_leaf_), cur_idx_(other.cur_idx_) {
            initialize();
        }

        ConstReverseIterator& operator=(const ConstReverseIterator& other) {
            if (this != &other) {
                cur_idx_ = other.cur_idx_;
                cur_leaf_ = other.cur_leaf_;
                cur_bitmap_idx_ = other.cur_bitmap_idx_;
                cur_bitmap_data_ = other.cur_bitmap_data_;
            }
            return *this;
        }

        ConstReverseIterator& operator++() {
            advance();
            return *this;
        }

        ConstReverseIterator operator++(int) {
            ConstReverseIterator tmp = *this;
            advance();
            return tmp;
        }
        // Does not return a reference because keys and payloads are stored separately.
        // If possible, use key() and payload() instead.
        Value operator*() const {
            return std::make_pair(cur_leaf_->key_slots_[cur_idx_], cur_leaf_->payload_slots_[cur_idx_]);
        }
        const double& key() const { return cur_leaf_->get_key(cur_idx_); }
        const double& payload() const { return cur_leaf_->get_payload(cur_idx_); }
        bool is_end() const { return cur_leaf_ == nullptr; }
        bool operator==(const ConstReverseIterator& rhs) const {
            return cur_idx_ == rhs.cur_idx_ && cur_leaf_ == rhs.cur_leaf_;
        }
        bool operator!=(const ConstReverseIterator& rhs) const {
            return !(*this == rhs);
        };

        private:
        void initialize() {
            if (!cur_leaf_) return;
            assert(cur_idx_ >= 0);
            if (cur_idx_ >= cur_leaf_->data_capacity_) {
              cur_leaf_ = cur_leaf_->next_leaf_;
              cur_idx_ = 0;
              if (!cur_leaf_) return;
            }
            cur_bitmap_idx_ = cur_idx_ >> 6;
            cur_bitmap_data_ = cur_leaf_->bitmap_[cur_bitmap_idx_];
            // Zero out extra bits
            int bit_pos = cur_idx_ - (cur_bitmap_idx_ << 6);
            cur_bitmap_data_ &= (1ULL << bit_pos) | ((1ULL << bit_pos) - 1);
            advance();
        }
        inline void advance() {
          while (cur_bitmap_data_ == 0) {
            cur_bitmap_idx_--;
            if (cur_bitmap_idx_ < 0) {
              cur_leaf_ = cur_leaf_->prev_leaf_;
              if (cur_leaf_ == nullptr) {
                cur_idx_ = 0;
                return;
              }
              cur_idx_ = cur_leaf_->data_capacity_ - 1;
              cur_bitmap_idx_ = cur_leaf_->bitmap_size_ - 1;
            }
            cur_bitmap_data_ = cur_leaf_->bitmap_[cur_bitmap_idx_];
          }
          assert(cpu_supports_bmi());
          int bit_pos = static_cast<int>(63 - _lzcnt_u64(cur_bitmap_data_));
          cur_idx_ = (cur_bitmap_idx_ << 6) + bit_pos;
          cur_bitmap_data_ &= ~(1ULL << bit_pos);
        }
    };

    // Iterates through all nodes with pre-order traversal
    class NodeIterator {
        public:
        const Alex* index_;
        AlexNode* cur_node_;
        std::stack<AlexNode*> node_stack_;  // helps with traversal

        // Start with root as cur and all children of root in stack
        NodeIterator(const Alex* index) : index_(index), cur_node_(index->root_node_) {
            if (cur_node_ && !cur_node_->is_leaf_) {
                auto node = static_cast<AlexModelNode*>(cur_node_);
                node_stack_.push(node->children_[node->num_children_ - 1]);
                for (int i = node->num_children_ - 2; i >= 0; i--) {
                    if (node->children_[i] != node->children_[i + 1]) {
                        node_stack_.push(node->children_[i]);
                    }
                }
            }
        }
        AlexNode* current() const { return cur_node_; }
        AlexNode* next() {
            if (node_stack_.empty()) {
                cur_node_ = nullptr;
                return nullptr;
            }
            cur_node_ = node_stack_.top();
            node_stack_.pop();

            if (!cur_node_->is_leaf_) {
                auto node = static_cast<AlexModelNode*>(cur_node_);
                node_stack_.push(node->children_[node->num_children_ - 1]);
                for (int i = node->num_children_ - 2; i >= 0; i--) {
                    if (node->children_[i] != node->children_[i + 1]) {
                        node_stack_.push(node->children_[i]);
                    }
                }
            }
            return cur_node_;
        }
        bool is_end() const { return cur_node_ == nullptr; }
    };
    
    /*** Operators ***/
    Alex& operator=(const Alex& other) {
        if (this != &other) {
            for (NodeIterator node_it = NodeIterator(this); !node_it.is_end(); node_it.next()) {
                delete_node(node_it.current());
            }
            delete_node(superroot_);
            derived_params_ = other.derived_params_;
            experimental_params_ = other.experimental_params_;
            istats_ = other.istats_;
            stats_ = other.stats_;
            key_less_ = other.key_less_;
            allocator_ = other.allocator_;
            superroot_ =
                static_cast<AlexModelNode*>(copy_tree_recursive(other.superroot_));
            root_node_ = superroot_->children_[0];
        }
        return *this;
    }
    Alex() {
        // Set up root as empty data node
        auto empty_data_node = new (AlexDataNode::alloc_type(allocator_).allocate(1)) AlexDataNode(key_less_, allocator_);
        empty_data_node->bulk_load(nullptr, 0);
        root_node_ = empty_data_node;
        create_superroot();
    }
    Alex(const Alex& other) : derived_params_(other.derived_params_),       stats_(other.stats_),
                         experimental_params_(other.experimental_params_), istats_(other.istats_),
                                            key_less_(other.key_less_), allocator_(other.allocator_) {
        superroot_ = static_cast<AlexModelNode*>(copy_tree_recursive(other.superroot_));
        root_node_ = superroot_->children_[0];
    }
    Alex(const AlexCompare& comp,
         const std::allocator<std::pair<double, double>>& alloc = std::allocator<std::pair<double, double>>())
        : key_less_(comp), allocator_(alloc) {
        // Set up root as empty data node
        auto empty_data_node = new (AlexDataNode::alloc_type(allocator_).allocate(1)) AlexDataNode(key_less_, allocator_);
        empty_data_node->bulk_load(nullptr, 0);
        root_node_ = empty_data_node;
        create_superroot();
    }
    Alex(const std::allocator<std::pair<double, double>>& alloc) : allocator_(alloc) {
        // Set up root as empty data node
        auto empty_data_node = new (AlexDataNode::alloc_type(allocator_).allocate(1)) AlexDataNode(key_less_, allocator_);
        empty_data_node->bulk_load(nullptr, 0);
        root_node_ = empty_data_node;
        create_superroot();
    }
    ~Alex() {
        for (NodeIterator node_it = NodeIterator(this); !node_it.is_end(); node_it.next()) {
            delete_node(node_it.current());
        }
        delete_node(superroot_);
    }

    // Return the data node that contains the key (if it exists).
    // Also optionally return the traversal path to the data node.
    // traversal_path should be empty when calling this function.
    // The returned traversal path begins with superroot and ends with the data node's parent.
    AlexDataNode* get_leaf(double key, std::vector<TraversalNode>* traversal_path = nullptr) const {
        if (traversal_path) {
            traversal_path->push_back({superroot_, 0});
        }
        AlexNode* cur = root_node_;

        while (!cur->is_leaf_) {
            auto node = static_cast<AlexModelNode*>(cur);
            int bucketID = node->model_.predict(key);
            bucketID = std::min<int>(std::max<int>(bucketID, 0), node->num_children_ - 1);
            if (traversal_path) {
                traversal_path->push_back({node, bucketID});
            }
            cur = node->children_[bucketID];
        }
        stats_.num_node_lookups += cur->level_;
        return static_cast<AlexDataNode*>(cur);
    }

    private:
    // Make a correction to the traversal path to instead point to the leaf node
    // that is to the left or right of the current leaf node.
    inline void correct_traversal_path(AlexDataNode* leaf, std::vector<TraversalNode>& traversal_path, bool left) const {
        if (left) {
            int repeats = 1 << leaf->duplication_factor_;
            TraversalNode& tn = traversal_path.back();
            AlexModelNode* parent = tn.node;
            // First bucket whose pointer is to leaf
            int start_bucketID = tn.bucketID - (tn.bucketID % repeats);
            if (start_bucketID == 0) {
                // Traverse back up the traversal path to make correction
                while (start_bucketID == 0) {
                    traversal_path.pop_back();
                    repeats = 1 << parent->duplication_factor_;
                    tn = traversal_path.back();
                    parent = tn.node;
                    start_bucketID = tn.bucketID - (tn.bucketID % repeats);
                }
                int correct_bucketID = start_bucketID - 1;
                tn.bucketID = correct_bucketID;
                AlexNode* cur = parent->children_[correct_bucketID];
                while (!cur->is_leaf_) {
                    auto node = static_cast<AlexModelNode*>(cur);
                    traversal_path.push_back({node, node->num_children_ - 1});
                    cur = node->children_[node->num_children_ - 1];
                }
                assert(cur == leaf->prev_leaf_);
            }
            else {
                tn.bucketID = start_bucketID - 1;
            }
        }
        else {
            int repeats = 1 << leaf->duplication_factor_;
            TraversalNode& tn = traversal_path.back();
            AlexModelNode* parent = tn.node;
            // First bucket whose pointer is not to leaf
            int end_bucketID = tn.bucketID - (tn.bucketID % repeats) + repeats;
            if (end_bucketID == parent->num_children_) {
                // Traverse back up the traversal path to make correction
                while (end_bucketID == parent->num_children_) {
                    traversal_path.pop_back();
                    repeats = 1 << parent->duplication_factor_;
                    tn = traversal_path.back();
                    parent = tn.node;
                    end_bucketID = tn.bucketID - (tn.bucketID % repeats) + repeats;
                }
                int correct_bucketID = end_bucketID;
                tn.bucketID = correct_bucketID;
                AlexNode* cur = parent->children_[correct_bucketID];
                while (!cur->is_leaf_) {
                    auto node = static_cast<AlexModelNode*>(cur);
                    traversal_path.push_back({node, 0});
                    cur = node->children_[0];
                }
                assert(cur == leaf->next_leaf_);
            }
            else {
                tn.bucketID = end_bucketID;
            }
        }
    }
    // Return left-most data node
    AlexDataNode* first_data_node() const {
        AlexNode* cur = root_node_;
        while (!cur->is_leaf_) {
            cur = static_cast<AlexModelNode*>(cur)->children_[0];
        }
        return static_cast<AlexDataNode*>(cur);
    }
    // Return right-most data node
    AlexDataNode* last_data_node() const {
        AlexNode* cur = root_node_;
        while (!cur->is_leaf_) {
            auto node = static_cast<AlexModelNode*>(cur);
            cur = node->children_[node->num_children_ - 1];
        }
        return static_cast<AlexDataNode*>(cur);
    }
    // Returns minimum key in the index
    double get_min_key() const { return first_data_node()->first_key(); }
    // Returns maximum key in the index
    double get_max_key() const { return last_data_node()->last_key(); }
    // Link all data nodes together. Used after bulk loading.
    void link_all_data_nodes() {
        AlexDataNode* prev_leaf = nullptr;
        for (NodeIterator node_it = NodeIterator(this); !node_it.is_end(); node_it.next()) {
            AlexNode* cur = node_it.current();
            if (cur->is_leaf_) {
                auto node = static_cast<AlexDataNode*>(cur);
                if (prev_leaf != nullptr) {
                    prev_leaf->next_leaf_ = node;
                    node->prev_leaf_ = prev_leaf;
                }
                prev_leaf = node;
            }
        }
    }
    // Link the new data nodes together when old data node is replaced by two new data nodes.
    void link_data_nodes(const AlexDataNode* old_leaf, AlexDataNode* left_leaf, AlexDataNode* right_leaf) {
        if (old_leaf->prev_leaf_ != nullptr) {
            old_leaf->prev_leaf_->next_leaf_ = left_leaf;
        }
        left_leaf->prev_leaf_ = old_leaf->prev_leaf_;
        left_leaf->next_leaf_ = right_leaf;
        right_leaf->prev_leaf_ = left_leaf;
        right_leaf->next_leaf_ = old_leaf->next_leaf_;
        if (old_leaf->next_leaf_ != nullptr) {
            old_leaf->next_leaf_->prev_leaf_ = right_leaf;
        }
    }

    void delete_node(AlexNode* node) {
        if (node == nullptr) {
            return;
        }
        else if (node->is_leaf_) {
            AlexDataNode::alloc_type(allocator_).destroy(static_cast<AlexDataNode*>(node));
            AlexDataNode::alloc_type(allocator_).deallocate(static_cast<AlexDataNode*>(node), 1);
        }
        else {
            AlexModelNode::alloc_type(allocator_).destroy(static_cast<AlexModelNode*>(node));
            AlexModelNode::alloc_type(allocator_).deallocate(static_cast<AlexModelNode*>(node), 1);
        }
    }
    // True if a == b
    inline bool key_equal(const double& a, const double& b) const {
        return !key_less_(a, b) && !key_less_(b, a);
    }

    public:
    // values should be the sorted array of key-payload pairs.
    // The number of elements should be num_keys.
    // The index must be empty when calling this method.
    void bulk_load(const Value values[], int num_keys) {
        if (stats_.num_keys > 0 || num_keys <= 0) {
            return;
        }
        delete_node(root_node_);  // delete the empty root node from constructor
        stats_.num_keys = num_keys;

        // Build temporary root model, which outputs a CDF in the range [0, 1]
        root_node_ = new (AlexModelNode::alloc_type(allocator_).allocate(1)) AlexModelNode(0, allocator_);
        double min_key = values[0].first;
        double max_key = values[num_keys - 1].first;
        root_node_->model_.a_ = 1.0 / (max_key - min_key);
        root_node_->model_.b_ = -1.0 * min_key * root_node_->model_.a_;

        // Compute cost of root node
        LinearModel<double> root_data_node_model;
        AlexDataNode::build_model(values, num_keys, &root_data_node_model);
        DataNodeStats stats;
        root_node_->cost_ = AlexDataNode::compute_expected_cost(values, num_keys, AlexDataNode::kInitDensity_, expected_insert_frac,
                                                                &root_data_node_model, &stats);
        // Recursively bulk load
        bulk_load_node(values, num_keys, root_node_, num_keys, &root_data_node_model);

        if (root_node_->is_leaf_) {
            static_cast<AlexDataNode*>(root_node_)->expected_avg_exp_search_iterations_ = stats.num_search_iterations;
            static_cast<AlexDataNode*>(root_node_)->expected_avg_shifts_ = stats.num_shifts;
        }
        create_superroot();
        update_superroot_key_domain();
        link_all_data_nodes();
    }
    private:
    // Only call this after creating a root node
    void create_superroot() {
        if (!root_node_) return;
        delete_node(superroot_);
        superroot_ = new (AlexModelNode::alloc_type(allocator_).allocate(1))
                          AlexModelNode(static_cast<short>(root_node_->level_ - 1), allocator_);
        superroot_->num_children_ = 1;
        superroot_->children_ = new (AlexModelNode::pointer_alloc_type(allocator_).allocate(1)) AlexNode*[1];
        update_superroot_pointer();
    }

    // Updates the key domain based on the min/max keys and retrains the model.
    // Should only be called immediately after bulk loading or when the root node is a data node.
    void update_superroot_key_domain() {
        assert(stats_.num_inserts == 0 || root_node_->is_leaf_);
        istats_.key_domain_min_ = get_min_key();
        istats_.key_domain_max_ = get_max_key();
        istats_.num_keys_at_last_right_domain_resize = stats_.num_keys;
        istats_.num_keys_at_last_left_domain_resize = stats_.num_keys;
        istats_.num_keys_above_key_domain = 0;
        istats_.num_keys_below_key_domain = 0;
        superroot_->model_.a_ = 1.0 / (istats_.key_domain_max_ - istats_.key_domain_min_);
        superroot_->model_.b_ = -1.0 * istats_.key_domain_min_ * superroot_->model_.a_;
    }
    void update_superroot_pointer() {
        superroot_->children_[0] = root_node_;
        superroot_->level_ = static_cast<short>(root_node_->level_ - 1);
    }

    // Recursively bulk load a single node.
    // Assumes node has already been trained to output [0, 1), has cost.
    // Figures out the optimal partitioning of children.
    // node is trained as if it's a model node.
    // data_node_model is what the node's model would be if it were a data node of dense keys.
    void bulk_load_node(const Value values[], int num_keys, AlexNode*& node, int total_keys,
                        const LinearModel<double>* data_node_model = nullptr) {
        // Automatically convert to data node when it is impossible to be better than current cost
        if (num_keys <= derived_params_.max_data_node_slots * AlexDataNode::kMinDensity_ &&
           (node->cost_ < kNodeLookupsWeight || node->model_.a_ == 0)) {
            stats_.num_data_nodes++;
            auto data_node = new (AlexDataNode::alloc_type(allocator_).allocate(1))
                                  AlexDataNode(node->level_, derived_params_.max_data_node_slots, key_less_, allocator_);
            data_node->bulk_load(values, num_keys, data_node_model);
            data_node->cost_ = node->cost_;
            delete_node(node);
            node = data_node;
            return;
        }

        // Use a fanout tree to determine the best way to divide the key space into
        // child nodes
        std::vector<fanout_tree::FTNode> used_fanout_tree_nodes;
        std::pair<int, double> best_fanout_stats;
        if (experimental_params_.fanout_selection_method == 0) {
            best_fanout_stats = fanout_tree::find_best_fanout_bottom_up<double, double>(
                                values, num_keys, node, total_keys, used_fanout_tree_nodes,
                                derived_params_.max_fanout, expected_insert_frac);
        }
        else if (experimental_params_.fanout_selection_method == 1) {
            best_fanout_stats = fanout_tree::find_best_fanout_top_down<double, double>(
                                values, num_keys, node, total_keys, used_fanout_tree_nodes,
                                derived_params_.max_fanout, expected_insert_frac);
        }
        int best_fanout_tree_depth = best_fanout_stats.first;
        double best_fanout_tree_cost = best_fanout_stats.second;

        // Decide whether this node should be a model node or data node
        if (best_fanout_tree_cost < node->cost_ || num_keys > derived_params_.max_data_node_slots * AlexDataNode::kMinDensity_) {
            // Convert to model node based on the output of the fanout tree
            stats_.num_model_nodes++;
            auto model_node = new (AlexModelNode::alloc_type(allocator_).allocate(1)) AlexModelNode(node->level_, allocator_);
            if (best_fanout_tree_depth == 0) {
                // slightly hacky: we assume this means that the node is relatively uniform but we need to split
                // to satisfy the max node size, so we compute the fanout that would satisfy that condition in expectation
                best_fanout_tree_depth =static_cast<int>(std::log2(static_cast<double>(num_keys) /
                                                                   derived_params_.max_data_node_slots)) + 1;
                used_fanout_tree_nodes.clear();
                fanout_tree::compute_level<double, double>(values, num_keys, node, total_keys, used_fanout_tree_nodes,
                                                           best_fanout_tree_depth, expected_insert_frac);
            }
            int fanout = 1 << best_fanout_tree_depth;
            model_node->model_.a_ = node->model_.a_ * fanout;
            model_node->model_.b_ = node->model_.b_ * fanout;
            model_node->num_children_ = fanout;
            model_node->children_ = new (AlexModelNode::pointer_alloc_type(allocator_).allocate(fanout)) AlexNode*[fanout];

            // Instantiate all the child nodes and recurse
            int cur = 0;
            for (fanout_tree::FTNode& tree_node : used_fanout_tree_nodes) {
                auto child_node = new (AlexModelNode::alloc_type(allocator_).allocate(1))
                                       AlexModelNode(static_cast<short>(node->level_ + 1), allocator_);
                child_node->cost_ = tree_node.cost;
                child_node->duplication_factor_ =
                    static_cast<uint8_t>(best_fanout_tree_depth - tree_node.level);
                int repeats = 1 << child_node->duplication_factor_;
                double left_value = static_cast<double>(cur) / fanout;
                double right_value = static_cast<double>(cur + repeats) / fanout;
                double left_boundary = (left_value - node->model_.b_) / node->model_.a_;
                double right_boundary =
                    (right_value - node->model_.b_) / node->model_.a_;
                child_node->model_.a_ = 1.0 / (right_boundary - left_boundary);
                child_node->model_.b_ = -child_node->model_.a_ * left_boundary;
                model_node->children_[cur] = child_node;
                LinearModel<double> child_data_node_model(tree_node.a, tree_node.b);
                bulk_load_node(values + tree_node.left_boundary, tree_node.right_boundary - tree_node.left_boundary,
                               model_node->children_[cur], total_keys, &child_data_node_model);
                model_node->children_[cur]->duplication_factor_ = static_cast<uint8_t>(best_fanout_tree_depth - tree_node.level);
                if (model_node->children_[cur]->is_leaf_) {
                    static_cast<AlexDataNode*>(model_node->children_[cur]) ->expected_avg_exp_search_iterations_ =
                        tree_node.expected_avg_search_iterations;
                    static_cast<AlexDataNode*>(model_node->children_[cur]) ->expected_avg_shifts_ =
                        tree_node.expected_avg_shifts;
                }
                for (int i = cur + 1; i < cur + repeats; i++) {
                  model_node->children_[i] = model_node->children_[cur];
                }
                cur += repeats;
            }
            delete_node(node);
            node = model_node;
        }
        else {
            // Convert to data node
            stats_.num_data_nodes++;
            auto data_node = new (AlexDataNode::alloc_type(allocator_).allocate(1))
                                  AlexDataNode(node->level_, derived_params_.max_data_node_slots, key_less_, allocator_);
            data_node->bulk_load(values, num_keys, data_node_model);
            data_node->cost_ = node->cost_;
            delete_node(node);
            node = data_node;
        }
    }

    // Caller needs to set the level, duplication factor, and neighbor pointers of the returned data node
    AlexDataNode* bulk_load_leaf_node_from_existing(AlexDataNode* existing_node, int left, int right,
                                                    bool compute_cost = true, const fanout_tree::FTNode* tree_node = nullptr,
                                                    bool reuse_model = false, bool keep_left = false, bool keep_right = false) {
        auto node = new (AlexDataNode::alloc_type(allocator_).allocate(1)) AlexDataNode(key_less_, allocator_);
        stats_.num_data_nodes++;
        if (tree_node) {
            // Use the model and num_keys saved in the tree node so we don't have to recompute it
            LinearModel<double> precomputed_model(tree_node->a, tree_node->b);
            node->bulk_load_from_existing(existing_node, left, right, keep_left, keep_right, &precomputed_model, tree_node->num_keys);
        }
        else if (reuse_model) {
            // Use the model from the existing node
            // Assumes the model is accurate
            int num_actual_keys = existing_node->num_keys_in_range(left, right);
            LinearModel<double> precomputed_model(existing_node->model_);
            precomputed_model.b_ -= left;
            precomputed_model.expand(static_cast<double>(num_actual_keys) / (right - left));
            node->bulk_load_from_existing(existing_node, left, right, keep_left, keep_right, &precomputed_model, num_actual_keys);
        }
        else {
            node->bulk_load_from_existing(existing_node, left, right, keep_left, keep_right);
        }
        node->max_slots_ = derived_params_.max_data_node_slots;
        if (compute_cost) {
            node->cost_ = node->compute_expected_cost(existing_node->frac_inserts());
        }
        return node;
    }

    /*** Lookup ***/
    public:
    // Looks for an exact match of the key
    // If the key does not exist, returns an end iterator
    // If there are multiple keys with the same value, returns an iterator to the right-most key
    // If you instead want an iterator to the left-most key with the input value, use lower_bound()
    typename Alex::Iterator find(const double& key) {
        stats_.num_lookups++;
        AlexDataNode* leaf = get_leaf(key);
        int idx = leaf->find_key(key);
        if (idx < 0) {
            return end();
        }
        else {
            return Iterator(leaf, idx);
        }
    }
    typename Alex::ConstIterator find(const double& key) const {
        stats_.num_lookups++;
        AlexDataNode* leaf = get_leaf(key);
        int idx = leaf->find_key(key);
        if (idx < 0) {
            return cend();
        }
        else {
            return ConstIterator(leaf, idx);
        }
    }
    size_t count(const double& key) {
        ConstIterator it = lower_bound(key);
        size_t num_equal = 0;
        while (!it.is_end() && key_equal(it.key(), key)) {
            num_equal++;
            ++it;
        }
        return num_equal;
    }
    // Returns an iterator to the first key no less than the input value
    typename Alex::Iterator lower_bound(const double& key) {
        stats_.num_lookups++;
        AlexDataNode* leaf = get_leaf(key);
        int idx = leaf->find_lower(key);
        return Iterator(leaf, idx);  // automatically handles the case where idx == leaf->data_capacity
    }
    typename Alex::ConstIterator lower_bound(const double& key) const {
        stats_.num_lookups++;
        AlexDataNode* leaf = get_leaf(key);
        int idx = leaf->find_lower(key);
        return ConstIterator(leaf, idx);  // automatically handles the case where idx == leaf->data_capacity
    }
    // Returns an iterator to the first key greater than the input value
    typename Alex::Iterator upper_bound(const double& key) {
        stats_.num_lookups++;
        AlexDataNode* leaf = get_leaf(key);
        int idx = leaf->find_upper(key);
        return Iterator(leaf, idx);  // automatically handles the case where idx == leaf->data_capacity
    }
    typename Alex::ConstIterator upper_bound(const double& key) const {
        stats_.num_lookups++;
        AlexDataNode* leaf = get_leaf(key);
        int idx = leaf->find_upper(key);
        return ConstIterator(leaf, idx);  // automatically handles the case where idx == leaf->data_capacity
    }

    std::pair<Iterator, Iterator> equal_range(const double& key) {
        return std::pair<Iterator, Iterator>(lower_bound(key), upper_bound(key));
    }
    std::pair<ConstIterator, ConstIterator> equal_range(const double& key) const {
        return std::pair<ConstIterator, ConstIterator>(lower_bound(key), upper_bound(key));
    }

    // Directly returns a pointer to the payload found through find(key)
    // This avoids the overhead of creating an iterator
    // Returns null pointer if there is no exact match of the key
    double* get_payload(const double& key) const {
        stats_.num_lookups++;
        AlexDataNode* leaf = get_leaf(key);
        int idx = leaf->find_key(key);
        if (idx < 0) {
            return nullptr;
        }
        else {
            return &(leaf->get_payload(idx));
        }
    }

    // Looks for the last key no greater than the input value
    // Conceptually, this is equal to the last key before upper_bound()
    typename Alex::Iterator find_last_no_greater_than(const double& key) {
        stats_.num_lookups++;
        AlexDataNode* leaf = get_leaf(key);
        int idx = leaf->upper_bound(key) - 1;
        if (idx == -1) {
            if (leaf->prev_leaf_) {
                // Edge case: need to check previous data node
                AlexDataNode* prev_leaf = leaf->prev_leaf_;
                int last_pos = prev_leaf->last_pos();
                return Iterator(prev_leaf, last_pos);
            }
            else {
                return Iterator(leaf, 0);
            }
        }
        else {
            return Iterator(leaf, idx);
        }
    }

    // Directly returns a pointer to the payload found through find_last_no_greater_than(key)
    // This avoids the overhead of creating an iterator
    double* get_payload_last_no_greater_than(const double& key) {
        stats_.num_lookups++;
        AlexDataNode* leaf = get_leaf(key);
        int idx = leaf->upper_bound(key) - 1;
        if (idx == -1) {
            if (leaf->prev_leaf_) {
                // Edge case: need to check previous data node
                AlexDataNode* prev_leaf = leaf->prev_leaf_;
                return &(prev_leaf->get_payload(prev_leaf->last_pos()));
            }
            else {
                return &(leaf->get_payload(leaf->first_pos()));
            }
        }
        else {
            return &(leaf->get_payload(idx));
        }
    }

    typename Alex::Iterator begin() {
        AlexNode* cur = root_node_;
        while (!cur->is_leaf_) {
            cur = static_cast<AlexModelNode*>(cur)->children_[0];
        }
        return Iterator(static_cast<AlexDataNode*>(cur), 0);
    }
    typename Alex::Iterator end() {
        Iterator it = Iterator();
        it.cur_leaf_ = nullptr;
        it.cur_idx_ = 0;
        return it;
    }
    typename Alex::ConstIterator cbegin() const {
        AlexNode* cur = root_node_;
        while (!cur->is_leaf_) {
            cur = static_cast<AlexModelNode*>(cur)->children_[0];
        }
        return ConstIterator(static_cast<AlexDataNode*>(cur), 0);
    }
    typename Alex::ConstIterator cend() const {
        ConstIterator it = ConstIterator();
        it.cur_leaf_ = nullptr;
        it.cur_idx_ = 0;
        return it;
    }
    typename Alex::ReverseIterator rbegin() {
        AlexNode* cur = root_node_;
        while (!cur->is_leaf_) {
            auto model_node = static_cast<AlexModelNode*>(cur);
            cur = model_node->children_[model_node->num_children_ - 1];
        }
        auto data_node = static_cast<AlexDataNode*>(cur);
        return ReverseIterator(data_node, data_node->data_capacity_ - 1);
    }
    typename Alex::ReverseIterator rend() {
        ReverseIterator it = ReverseIterator();
        it.cur_leaf_ = nullptr;
        it.cur_idx_ = 0;
        return it;
    }
    typename Alex::ConstReverseIterator crbegin() const {
        AlexNode* cur = root_node_;
        while (!cur->is_leaf_) {
            auto model_node = static_cast<AlexModelNode*>(cur);
            cur = model_node->children_[model_node->num_children_ - 1];
        }
        auto data_node = static_cast<AlexDataNode*>(cur);
        return ConstReverseIterator(data_node, data_node->data_capacity_ - 1);
    }
    typename Alex::ConstReverseIterator crend() const {
        ConstReverseIterator it = ConstReverseIterator();
        it.cur_leaf_ = nullptr;
        it.cur_idx_ = 0;
        return it;
    }

    /*** Insert ***/
    std::pair<Iterator, bool> insert(const Value& value) {
        return insert(value.first, value.second);
    }

    template <class InputIterator>
    void insert(InputIterator first, InputIterator last) {
        for (auto it = first; it != last; ++it) {
            insert(*it);
        }
    }

    // This will NOT do an update of an existing key.
    // To perform an update or read-modify-write, do a lookup and modify the payload's value.
    // Returns iterator to inserted element, and whether the insert happened or not.
    // Insert does not happen if duplicates are not allowed and duplicate is found.
    std::pair<Iterator, bool> insert(const double& key, const double& payload) {
        // If enough keys fall outside the key domain, expand the root to expand the key domain
        if (key > istats_.key_domain_max_) {
            istats_.num_keys_above_key_domain++;
            if (should_expand_right()) {
                expand_root(key, false);  // expand to the right
            }
        }
        else if (key < istats_.key_domain_min_) {
            istats_.num_keys_below_key_domain++;
            if (should_expand_left()) {
                expand_root(key, true);  // expand to the left
            }
        }
        AlexDataNode* leaf = get_leaf(key);

        // Nonzero fail flag means that the insert did not happen
        std::pair<int, int> ret = leaf->insert(key, payload);
        int fail = ret.first;
        int insert_pos = ret.second;
        if (fail == -1) {
            // Duplicate found and duplicates not allowed
            return {Iterator(leaf, insert_pos), false};
        }
        // If no insert, figure out what to do with the data node to decrease the cost
        if (fail) {
            std::vector<TraversalNode> traversal_path;
            get_leaf(key, &traversal_path);
            AlexModelNode* parent = traversal_path.back().node;

            while (fail) {
                auto start_time = std::chrono::high_resolution_clock::now();
                stats_.num_expand_and_scales += leaf->num_resizes_;

                if (parent == superroot_) {
                    update_superroot_key_domain();
                }
                int bucketID = parent->model_.predict(key);
                bucketID = std::min<int>(std::max<int>(bucketID, 0), parent->num_children_ - 1);
                std::vector<fanout_tree::FTNode> used_fanout_tree_nodes;

                int fanout_tree_depth = 1;
                if (experimental_params_.splitting_policy_method == 0 || fail >= 2) {
                    // always split in 2. No extra work required here
                }
                else if (experimental_params_.splitting_policy_method == 1) {
                    // decide between no split (i.e., expand and retrain) or splitting in 2
                    fanout_tree_depth = fanout_tree::find_best_fanout_existing_node<double, double>(
                                        parent, bucketID, stats_.num_keys, used_fanout_tree_nodes, 2);
                }
                else if (experimental_params_.splitting_policy_method == 2) {
                    // use full fanout tree to decide fanout
                    fanout_tree_depth = fanout_tree::find_best_fanout_existing_node<double, double>(
                                        parent, bucketID, stats_.num_keys, used_fanout_tree_nodes, derived_params_.max_fanout);
                }
                int best_fanout = 1 << fanout_tree_depth;
                stats_.cost_computation_time += std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                std::chrono::high_resolution_clock::now() - start_time).count();

                if (fanout_tree_depth == 0) {
                    // expand existing data node and retrain model
                    leaf->resize(AlexDataNode::kMinDensity_, true, leaf->is_append_mostly_right(), leaf->is_append_mostly_left());
                    fanout_tree::FTNode& tree_node = used_fanout_tree_nodes[0];
                    leaf->cost_ = tree_node.cost;
                    leaf->expected_avg_exp_search_iterations_ = tree_node.expected_avg_search_iterations;
                    leaf->expected_avg_shifts_ = tree_node.expected_avg_shifts;
                    leaf->reset_stats();
                    stats_.num_expand_and_retrains++;
                }
                else {
                    // split data node: always try to split sideways/upwards, only split downwards if necessary
                    bool reuse_model = (fail == 3);
                    if (experimental_params_.allow_splitting_upwards) {
                        // allow splitting upwards
                        assert(experimental_params_.splitting_policy_method != 2);
                        int stop_propagation_level = best_split_propagation(traversal_path);
                        if (stop_propagation_level <= superroot_->level_) {
                            parent = split_downwards(parent, bucketID, fanout_tree_depth, used_fanout_tree_nodes, reuse_model);
                        }
                        else {
                            split_upwards(key, stop_propagation_level, traversal_path, reuse_model, &parent);
                        }
                    }
                    else {
                        // either split sideways or downwards
                        bool should_split_downwards = (parent->num_children_ * best_fanout / (1 << leaf->duplication_factor_)
                                                      > derived_params_.max_fanout || parent->level_ == superroot_->level_);
                        if (should_split_downwards) {
                            parent = split_downwards(parent, bucketID, fanout_tree_depth, used_fanout_tree_nodes, reuse_model);
                        }
                        else {
                            split_sideways(parent, bucketID, fanout_tree_depth, used_fanout_tree_nodes, reuse_model);
                        }
                    }
                    leaf = static_cast<AlexDataNode*>(parent->get_child_node(key));
                }
                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration = end_time - start_time;
                stats_.splitting_time += std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();

                // Try again to insert the key
                ret = leaf->insert(key, payload);
                fail = ret.first;
                insert_pos = ret.second;
                if (fail == -1) {
                    // Duplicate found and duplicates not allowed
                    return {Iterator(leaf, insert_pos), false};
                }
            }
        }
        stats_.num_inserts++;
        stats_.num_keys++;
        return {Iterator(leaf, insert_pos), true};
    }

    private:
    // Our criteria for when to expand the root, thereby expanding the key domain.
    // We want to strike a balance between expanding too aggressively and too slowly.
    // Specifically, the number of inserted keys falling to the right of the key domain must have one of two properties:
    // (1) above some maximum threshold, or
    // (2) above some minimum threshold and the number is much more than we would expect from randomness alone.
    bool should_expand_right() const {
        return (!root_node_->is_leaf_ &&
                ((istats_.num_keys_above_key_domain >= kMinOutOfDomainKeys &&
                  istats_.num_keys_above_key_domain >=
                    kOutOfDomainToleranceFactor * (stats_.num_keys / istats_.num_keys_at_last_right_domain_resize -1)) ||
                  istats_.num_keys_above_key_domain >= kMaxOutOfDomainKeys));
    }
    // Similar to should_expand_right, but for insertions to the left of the key domain.
    bool should_expand_left() const {
        return (!root_node_->is_leaf_ &&
                ((istats_.num_keys_below_key_domain >= kMinOutOfDomainKeys &&
                  istats_.num_keys_below_key_domain >=
                    kOutOfDomainToleranceFactor * (stats_.num_keys / istats_.num_keys_at_last_left_domain_resize - 1)) ||
                  istats_.num_keys_below_key_domain >= kMaxOutOfDomainKeys));
    }

    // When splitting upwards, find best internal node to propagate upwards to.
    // Returns the level of that node. Returns superroot's level if splitting sideways not possible.
    int best_split_propagation(const std::vector<TraversalNode>& traversal_path, bool verbose = false) const {
        if (root_node_->is_leaf_) {
            return superroot_->level_;
        }
        // Find costs on the path down to data node
        std::vector<SplitDecisionCosts> traversal_costs;
        for (const TraversalNode& tn : traversal_path) {
            double stop_cost;
            AlexNode* next = tn.node->children_[tn.bucketID];
            if (next->duplication_factor_ > 0) {
                stop_cost = 0;
            }
            else {
                stop_cost = tn.node->num_children_ >= derived_params_.max_fanout ? std::numeric_limits<double>::max()
                                                        : tn.node->num_children_ + SplitDecisionCosts::base_cost;
            }
            traversal_costs.push_back({stop_cost,
                            tn.node->num_children_ <= 2 ? 0 : (tn.node->num_children_ >> 1) + SplitDecisionCosts::base_cost});
        }
        // Compute back upwards to find the optimal node to stop propagation.
        // Ignore the superroot (the first node in the traversal path).
        double cumulative_cost = 0;
        double best_cost = std::numeric_limits<double>::max();
        int best_path_level = superroot_->level_;
        for (int i = traversal_costs.size() - 1; i >= 0; i--) {
            SplitDecisionCosts& c = traversal_costs[i];
            if (c.stop_cost != std::numeric_limits<double>::max() && cumulative_cost + c.stop_cost < best_cost) {
                best_cost = cumulative_cost + c.stop_cost;
                best_path_level = traversal_path[i].node->level_;
            }
            cumulative_cost += c.split_cost;
        }
        return best_path_level;
    }

    // Expand the key value space that is covered by the index.
    // Expands the root node (which is a model node).
    // If the root node is at the max node size, then we split the root and create a new root node.
    void expand_root(double key, bool expand_left) {
        auto root = static_cast<AlexModelNode*>(root_node_);
        // Find the new bounds of the key domain.
        // Need to be careful to avoid overflows in the key type.
        double domain_size = istats_.key_domain_max_ - istats_.key_domain_min_;
        int expansion_factor;
        double new_domain_min = istats_.key_domain_min_;
        double new_domain_max = istats_.key_domain_max_;
        AlexDataNode* outermost_node;
        if (expand_left) {
            auto key_difference = static_cast<double>(istats_.key_domain_min_ -
                                                        std::min(key, get_min_key()));
            expansion_factor = pow_2_round_up(static_cast<int>(
                std::ceil((key_difference + domain_size) / domain_size)));
            // Check for overflow. To avoid overflow on signed types while doing
            // this check, we do comparisons using half of the relevant quantities.
            double half_expandable_domain =
                istats_.key_domain_max_ / 2 - std::numeric_limits<double>::lowest() / 2;
            double half_expanded_domain_size = expansion_factor / 2 * domain_size;
            if (half_expandable_domain < half_expanded_domain_size) {
                new_domain_min = std::numeric_limits<double>::lowest();
            }
            else {
                new_domain_min = istats_.key_domain_max_;
                new_domain_min -= half_expanded_domain_size;
                new_domain_min -= half_expanded_domain_size;
            }
            istats_.num_keys_at_last_left_domain_resize = stats_.num_keys;
            istats_.num_keys_below_key_domain = 0;
            outermost_node = first_data_node();
        }
        else {
            auto key_difference = static_cast<double>(std::max(key, get_max_key()) - istats_.key_domain_max_);
            expansion_factor = pow_2_round_up(static_cast<int>(std::ceil((key_difference + domain_size) / domain_size)));
            // Check for overflow. To avoid overflow on signed types while doing this check,
            // we do comparisons using half of the relevant quantities.
            double half_expandable_domain = std::numeric_limits<double>::max() / 2 - istats_.key_domain_min_ / 2;
            double half_expanded_domain_size = expansion_factor / 2 * domain_size;
            if (half_expandable_domain < half_expanded_domain_size) {
                new_domain_max = std::numeric_limits<double>::max();
            }
            else {
                new_domain_max = istats_.key_domain_min_;
                new_domain_max += half_expanded_domain_size;
                new_domain_max += half_expanded_domain_size;
            }
            istats_.num_keys_at_last_right_domain_resize = stats_.num_keys;
            istats_.num_keys_above_key_domain = 0;
            outermost_node = last_data_node();
        }
        assert(expansion_factor > 1);

        // Modify the root node appropriately
        int new_nodes_start;  // index of first pointer to a new node
        int new_nodes_end;    // exclusive
        if (root->num_children_ * expansion_factor <= derived_params_.max_fanout) {
            // Expand root node
            stats_.num_model_node_expansions++;
            stats_.num_model_node_expansion_pointers += root->num_children_;

            int new_num_children = root->num_children_ * expansion_factor;
            auto new_children = new (AlexModelNode::pointer_alloc_type(allocator_).allocate(new_num_children))
                                     AlexNode*[new_num_children];
            int copy_start;
            if (expand_left) {
                copy_start = new_num_children - root->num_children_;
                new_nodes_start = 0;
                new_nodes_end = copy_start;
                root->model_.b_ += new_num_children - root->num_children_;
            }
            else {
                copy_start = 0;
                new_nodes_start = root->num_children_;
                new_nodes_end = new_num_children;
            }
            for (int i = 0; i < root->num_children_; i++) {
                new_children[copy_start + i] = root->children_[i];
            }
            AlexModelNode::pointer_alloc_type(allocator_).deallocate(root->children_, root->num_children_);
            root->children_ = new_children;
            root->num_children_ = new_num_children;
        }
        else {
            // Create new root node
            auto new_root = new (AlexModelNode::alloc_type(allocator_).allocate(1))
                                 AlexModelNode(static_cast<short>(root->level_ - 1), allocator_);
            new_root->model_.a_ = root->model_.a_;
            if (expand_left) {
                new_root->model_.b_ = root->model_.b_ + expansion_factor - 1;
            }
            else {
                new_root->model_.b_ = root->model_.b_;
            }
            new_root->num_children_ = expansion_factor;
            new_root->children_ = new (AlexModelNode::pointer_alloc_type(allocator_).allocate(expansion_factor))
                                       AlexNode*[expansion_factor];
            if (expand_left) {
                new_root->children_[expansion_factor - 1] = root;
                new_nodes_start = 0;
            }
            else {
                new_root->children_[0] = root;
                new_nodes_start = 1;
            }
            new_nodes_end = new_nodes_start + expansion_factor - 1;
            root_node_ = new_root;
            update_superroot_pointer();
            root = new_root;
        }

        // Determine if new nodes represent a range outside the key type's domain.
        // This happens when we're preventing overflows.
        int in_bounds_new_nodes_start = new_nodes_start;
        int in_bounds_new_nodes_end = new_nodes_end;
        if (expand_left) {
            in_bounds_new_nodes_start = std::max(new_nodes_start, root->model_.predict(new_domain_min));
        }
        else {
            in_bounds_new_nodes_end = std::min(new_nodes_end, root->model_.predict(new_domain_max) + 1);
        }

        // Fill newly created child pointers of the root node with new data nodes.
        // To minimize empty new data nodes, we create a new data node per n child pointers,
        // where n is the number of pointers to existing nodes.
        // Requires reassigning some keys from the outermost pre-existing data node to the new data nodes.
        int n = root->num_children_ - (new_nodes_end - new_nodes_start);
        assert(root->num_children_ % n == 0);
        auto new_node_duplication_factor = static_cast<uint8_t>(log_2_round_down(n));
        if (expand_left) {
            double left_boundary_value = istats_.key_domain_min_;
            int left_boundary = outermost_node->lower_bound(left_boundary_value);
            AlexDataNode* next = outermost_node;
            for (int i = new_nodes_end; i > new_nodes_start; i -= n) {
                if (i <= in_bounds_new_nodes_start) {
                    // Do not initialize nodes that fall outside the key type's domain
                    break;
                }
                int right_boundary = left_boundary;
                if (i - n <= in_bounds_new_nodes_start) {
                    left_boundary = 0;
                }
                else {
                    left_boundary_value -= domain_size;
                    left_boundary = outermost_node->lower_bound(left_boundary_value);
                }
                AlexDataNode* new_node = bulk_load_leaf_node_from_existing(outermost_node, left_boundary, right_boundary, true);
                new_node->level_ = static_cast<short>(root->level_ + 1);
                new_node->duplication_factor_ = new_node_duplication_factor;
                if (next) {
                    next->prev_leaf_ = new_node;
                }
                new_node->next_leaf_ = next;
                next = new_node;
                for (int j = i - 1; j >= i - n; j--) {
                    root->children_[j] = new_node;
                }
            }
        }
        else {
            double right_boundary_value = istats_.key_domain_max_;
            int right_boundary = outermost_node->lower_bound(right_boundary_value);
            AlexDataNode* prev = nullptr;
            for (int i = new_nodes_start; i < new_nodes_end; i += n) {
                if (i >= in_bounds_new_nodes_end) {
                    // Do not initialize nodes that fall outside the key type's domain
                    break;
                }
                int left_boundary = right_boundary;
                if (i + n >= in_bounds_new_nodes_end) {
                    right_boundary = outermost_node->data_capacity_;
                }
                else {
                    right_boundary_value += domain_size;
                    right_boundary = outermost_node->lower_bound(right_boundary_value);
                }
                AlexDataNode* new_node = bulk_load_leaf_node_from_existing(outermost_node, left_boundary, right_boundary, true);
                new_node->level_ = static_cast<short>(root->level_ + 1);
                new_node->duplication_factor_ = new_node_duplication_factor;
                if (prev) {
                    prev->next_leaf_ = new_node;
                }
                new_node->prev_leaf_ = prev;
                prev = new_node;
                for (int j = i; j < i + n; j++) {
                    root->children_[j] = new_node;
                }
            }
        }
        // Connect leaf nodes and remove reassigned keys from outermost pre-existing node.
        if (expand_left) {
            outermost_node->erase_range(new_domain_min, istats_.key_domain_min_);
            auto last_new_leaf = static_cast<AlexDataNode*>(root->children_[new_nodes_end - 1]);
            outermost_node->prev_leaf_ = last_new_leaf;
            last_new_leaf->next_leaf_ = outermost_node;
        }
        else {
            outermost_node->erase_range(istats_.key_domain_max_, new_domain_max, true);
            auto first_new_leaf = static_cast<AlexDataNode*>(root->children_[new_nodes_start]);
            outermost_node->next_leaf_ = first_new_leaf;
            first_new_leaf->prev_leaf_ = outermost_node;
        }
        istats_.key_domain_min_ = new_domain_min;
        istats_.key_domain_max_ = new_domain_max;
    }

    // Splits downwards in the manner determined by the fanout tree and updates the pointers of the parent.
    // If no fanout tree is provided, then splits downward in two. Returns the newly created model node.
    AlexModelNode* split_downwards(AlexModelNode* parent, int bucketID, int fanout_tree_depth,
                                   const std::vector<fanout_tree::FTNode>& used_fanout_tree_nodes, bool reuse_model) {
        auto leaf = static_cast<AlexDataNode*>(parent->children_[bucketID]);
        stats_.num_downward_splits++;
        stats_.num_downward_split_keys += leaf->num_keys_;

        // Create the new model node that will replace the current data node
        int fanout = 1 << fanout_tree_depth;
        auto new_node = new (AlexModelNode::alloc_type(allocator_).allocate(1)) AlexModelNode(leaf->level_, allocator_);
        new_node->duplication_factor_ = leaf->duplication_factor_;
        new_node->num_children_ = fanout;
        new_node->children_ = new (AlexModelNode::pointer_alloc_type(allocator_).allocate(fanout)) AlexNode*[fanout];

        int repeats = 1 << leaf->duplication_factor_;
        int start_bucketID = bucketID - (bucketID % repeats);  // first bucket with same child
        int end_bucketID = start_bucketID + repeats;  // first bucket with different child
        double left_boundary_value = (start_bucketID - parent->model_.b_) / parent->model_.a_;
        double right_boundary_value = (end_bucketID - parent->model_.b_) / parent->model_.a_;
        new_node->model_.a_ = 1.0 / (right_boundary_value - left_boundary_value) * fanout;
        new_node->model_.b_ = -new_node->model_.a_ * left_boundary_value;

        // Create new data nodes
        if (used_fanout_tree_nodes.empty()) {
            assert(fanout_tree_depth == 1);
            create_two_new_data_nodes(leaf, new_node, fanout_tree_depth, reuse_model);
        }
        else {
            create_new_data_nodes(leaf, new_node, fanout_tree_depth, used_fanout_tree_nodes);
        }

        delete_node(leaf);
        stats_.num_data_nodes--;
        stats_.num_model_nodes++;
        for (int i = start_bucketID; i < end_bucketID; i++) {
            parent->children_[i] = new_node;
        }
        if (parent == superroot_) {
            root_node_ = new_node;
            update_superroot_pointer();
        }
        return new_node;
    }

    // Splits data node sideways in the manner determined by the fanout tree.
    // If no fanout tree is provided, then splits sideways in two.
    void split_sideways(AlexModelNode* parent, int bucketID, int fanout_tree_depth,
                        const std::vector<fanout_tree::FTNode>& used_fanout_tree_nodes, bool reuse_model) {
        auto leaf = static_cast<AlexDataNode*>(parent->children_[bucketID]);
        stats_.num_sideways_splits++;
        stats_.num_sideways_split_keys += leaf->num_keys_;

        int fanout = 1 << fanout_tree_depth;
        int repeats = 1 << leaf->duplication_factor_;
        if (fanout > repeats) {
            // Expand the pointer array in the parent model node if there are not
            // enough redundant pointers
            stats_.num_model_node_expansions++;
            stats_.num_model_node_expansion_pointers += parent->num_children_;
            int expansion_factor = parent->expand(fanout_tree_depth - leaf->duplication_factor_);
            repeats *= expansion_factor;
            bucketID *= expansion_factor;
        }
        int start_bucketID = bucketID - (bucketID % repeats);  // first bucket with same child

        if (used_fanout_tree_nodes.empty()) {
            assert(fanout_tree_depth == 1);
            create_two_new_data_nodes(leaf, parent, std::max(fanout_tree_depth, static_cast<int>(leaf->duplication_factor_)),
                                      reuse_model, start_bucketID);
        }
        else {
            // Extra duplication factor is required when there are more redundant
            // pointers than necessary
            int extra_duplication_factor = std::max(0, leaf->duplication_factor_ - fanout_tree_depth);
            create_new_data_nodes(leaf, parent, fanout_tree_depth, used_fanout_tree_nodes, start_bucketID, extra_duplication_factor);
        }
        delete_node(leaf);
        stats_.num_data_nodes--;
    }

    // Create two new data nodes by equally dividing the key space of the old data node, insert the new
    // nodes as children of the parent model node starting from a given position, and link the new data nodes together.
    // duplication_factor denotes how many child pointer slots were assigned to the old data node.
    void create_two_new_data_nodes(AlexDataNode* old_node, AlexModelNode* parent, int duplication_factor, bool reuse_model,
                                   int start_bucketID = 0) {
        assert(duplication_factor >= 1);
        int num_buckets = 1 << duplication_factor;
        int end_bucketID = start_bucketID + num_buckets;
        int mid_bucketID = start_bucketID + (num_buckets >> 1);

        bool append_mostly_right = old_node->is_append_mostly_right();
        int appending_right_bucketID = std::min<int>(std::max<int>(parent->model_.predict(old_node->max_key_), 0),
                                                     parent->num_children_ - 1);
        bool append_mostly_left = old_node->is_append_mostly_left();
        int appending_left_bucketID = std::min<int>(std::max<int>(parent->model_.predict(old_node->min_key_), 0),
                                                    parent->num_children_ - 1);

        int right_boundary = old_node->lower_bound((mid_bucketID - parent->model_.b_) / parent->model_.a_);
        AlexDataNode* left_leaf = bulk_load_leaf_node_from_existing(old_node, 0, right_boundary, true, nullptr, reuse_model,
                                    append_mostly_right && start_bucketID <= appending_right_bucketID &&
                                                 appending_right_bucketID < mid_bucketID,
                                     append_mostly_left && start_bucketID <= appending_left_bucketID &&
                                                  appending_left_bucketID < mid_bucketID);
        AlexDataNode* right_leaf = bulk_load_leaf_node_from_existing(old_node, right_boundary, old_node->data_capacity_,
                                                                     true, nullptr, reuse_model,
                                    append_mostly_right && mid_bucketID <= appending_right_bucketID &&
                                               appending_right_bucketID < end_bucketID,
                                     append_mostly_left && mid_bucketID <= appending_left_bucketID &&
                                                appending_left_bucketID < end_bucketID);
        left_leaf->level_ = static_cast<short>(parent->level_ + 1);
        right_leaf->level_ = static_cast<short>(parent->level_ + 1);
        left_leaf->duplication_factor_ = static_cast<uint8_t>(duplication_factor - 1);
        right_leaf->duplication_factor_ = static_cast<uint8_t>(duplication_factor - 1);

        for (int i = start_bucketID; i < mid_bucketID; i++) {
            parent->children_[i] = left_leaf;
        }
        for (int i = mid_bucketID; i < end_bucketID; i++) {
            parent->children_[i] = right_leaf;
        }
        link_data_nodes(old_node, left_leaf, right_leaf);
    }

    // Create new data nodes from the keys in the old data node according to the fanout tree,
    // insert the new nodes as children of the parent model node starting from a given position, and link the new data nodes together.
    // Helper for splitting when using a fanout tree.
    void create_new_data_nodes(AlexDataNode* old_node, AlexModelNode* parent, int fanout_tree_depth,
                               const std::vector<fanout_tree::FTNode>& used_fanout_tree_nodes,
                               int start_bucketID = 0, int extra_duplication_factor = 0) {
        bool append_mostly_right = old_node->is_append_mostly_right();
        int appending_right_bucketID = std::min<int>(std::max<int>(parent->model_.predict(old_node->max_key_), 0),
                                                     parent->num_children_ - 1);
        bool append_mostly_left = old_node->is_append_mostly_left();
        int appending_left_bucketID = std::min<int>(std::max<int>(parent->model_.predict(old_node->min_key_), 0),
                                                    parent->num_children_ - 1);

        // Create the new data nodes
        int cur = start_bucketID;  // first bucket with same child
        AlexDataNode* prev_leaf = old_node->prev_leaf_;  // used for linking the new data nodes
        for (const fanout_tree::FTNode& tree_node : used_fanout_tree_nodes) {
            auto duplication_factor = static_cast<uint8_t>(fanout_tree_depth - tree_node.level + extra_duplication_factor);
            int child_node_repeats = 1 << duplication_factor;
            bool keep_left = append_mostly_right && cur <= appending_right_bucketID &&
                               appending_right_bucketID < cur + child_node_repeats;
            bool keep_right = append_mostly_left && cur <= appending_left_bucketID &&
                                appending_left_bucketID < cur + child_node_repeats;
            AlexDataNode* child_node = bulk_load_leaf_node_from_existing(old_node, tree_node.left_boundary, tree_node.right_boundary,
                                                                         false, &tree_node, false, keep_left, keep_right);
            child_node->level_ = static_cast<short>(parent->level_ + 1);
            child_node->cost_ = tree_node.cost;
            child_node->duplication_factor_ = duplication_factor;
            child_node->expected_avg_exp_search_iterations_ = tree_node.expected_avg_search_iterations;
            child_node->expected_avg_shifts_ = tree_node.expected_avg_shifts;
            child_node->prev_leaf_ = prev_leaf;
            if (prev_leaf != nullptr) {
                prev_leaf->next_leaf_ = child_node;
            }
            for (int i = cur; i < cur + child_node_repeats; i++) {
                parent->children_[i] = child_node;
            }
            cur += child_node_repeats;
            prev_leaf = child_node;
        }
        prev_leaf->next_leaf_ = old_node->next_leaf_;
        if (old_node->next_leaf_ != nullptr) {
            old_node->next_leaf_->prev_leaf_ = prev_leaf;
        }
    }

    // Splits the data node in two and propagates the split upwards along the
    // traversal path.
    // Of the two newly created data nodes, returns the one that key falls into.
    // Returns the parent model node of the new data nodes through new_parent.
    AlexDataNode* split_upwards(double key, int stop_propagation_level, const std::vector<TraversalNode>& traversal_path,
                                bool reuse_model, AlexModelNode** new_parent, bool verbose = false) {
        assert(stop_propagation_level >= root_node_->level_);
        std::vector<AlexNode*> to_delete;  // nodes that need to be deleted

        // Split the data node into two new data nodes
        const TraversalNode& parent_path_node = traversal_path.back();
        AlexModelNode* parent = parent_path_node.node;
        auto leaf = static_cast<AlexDataNode*>(parent->children_[parent_path_node.bucketID]);
        int leaf_repeats = 1 << (leaf->duplication_factor_);
        int leaf_start_bucketID = parent_path_node.bucketID - (parent_path_node.bucketID % leaf_repeats);
        double leaf_mid_bucketID = leaf_start_bucketID + leaf_repeats / 2.0;
        int leaf_end_bucketID = leaf_start_bucketID + leaf_repeats;  // first bucket with next child
        stats_.num_sideways_splits++;
        stats_.num_sideways_split_keys += leaf->num_keys_;

        // Determine if either of the two new data nodes will need to adapt to
        // append-mostly behavior
        bool append_mostly_right = leaf->is_append_mostly_right();
        bool left_half_appending_right = false, right_half_appending_right = false;
        if (append_mostly_right) {
            double appending_right_bucketID = parent->model_.predict_double(leaf->max_key_);
            if (appending_right_bucketID >= leaf_start_bucketID && appending_right_bucketID < leaf_mid_bucketID) {
                left_half_appending_right = true;
            }
            else if (appending_right_bucketID >= leaf_mid_bucketID && appending_right_bucketID < leaf_end_bucketID) {
                right_half_appending_right = true;
            }
        }
        bool append_mostly_left = leaf->is_append_mostly_left();
        bool left_half_appending_left = false, right_half_appending_left = false;
        if (append_mostly_left) {
            double appending_left_bucketID = parent->model_.predict_double(leaf->min_key_);
            if (appending_left_bucketID >= leaf_start_bucketID && appending_left_bucketID < leaf_mid_bucketID) {
                left_half_appending_left = true;
            }
            else if (appending_left_bucketID >= leaf_mid_bucketID && appending_left_bucketID < leaf_end_bucketID) {
                right_half_appending_left = true;
            }
        }

        int mid_boundary = leaf->lower_bound((leaf_mid_bucketID - parent->model_.b_) / parent->model_.a_);
        AlexDataNode* left_leaf = bulk_load_leaf_node_from_existing(leaf, 0, mid_boundary, true, nullptr,
                                                                    reuse_model,
                                                                    append_mostly_right && left_half_appending_right,
                                                                     append_mostly_left && left_half_appending_left);
        AlexDataNode* right_leaf = bulk_load_leaf_node_from_existing(leaf, mid_boundary, leaf->data_capacity_, true, nullptr,
                                                                     reuse_model,
                                                                     append_mostly_right && right_half_appending_right,
                                                                      append_mostly_left && right_half_appending_left);
        // This is the expected duplication factor; it will be correct once we
        // split/expand the parent
        left_leaf->duplication_factor_ = leaf->duplication_factor_;
        right_leaf->duplication_factor_ = leaf->duplication_factor_;
        left_leaf->level_ = leaf->level_;
        right_leaf->level_ = leaf->level_;
        link_data_nodes(leaf, left_leaf, right_leaf);
        to_delete.push_back(leaf);
        stats_.num_data_nodes--;

        // The new data node that the key falls into is the one we return
        AlexDataNode* new_data_node;
        if (parent->model_.predict_double(key) < leaf_mid_bucketID) {
            new_data_node = left_leaf;
        }
        else {
            new_data_node = right_leaf;
        }

        // Split all internal nodes from the parent up to the highest node along the traversal path.
        // As this happens, the entries of the traversal path will go stale, which is fine because we no longer use them.
        // Splitting an internal node involves dividing the child pointers into two halves, and doubling the relevant half.
        AlexNode* prev_left_split = left_leaf;
        AlexNode* prev_right_split = right_leaf;
        int path_idx = static_cast<int>(traversal_path.size()) - 1;
        while (traversal_path[path_idx].node->level_ > stop_propagation_level) {
            // Decide which half to double
            const TraversalNode& path_node = traversal_path[path_idx];
            AlexModelNode* cur_node = path_node.node;
            stats_.num_model_node_splits++;
            stats_.num_model_node_split_pointers += cur_node->num_children_;
            bool double_left_half = path_node.bucketID < (cur_node->num_children_ >> 1);
            AlexModelNode* left_split = nullptr;
            AlexModelNode* right_split = nullptr;

            // If one of the resulting halves will only have one child pointer, we
            // should "pull up" that child
            bool pull_up_left_child = false, pull_up_right_child = false;
            AlexNode* left_half_first_child = cur_node->children_[0];
            AlexNode* right_half_first_child = cur_node->children_[cur_node->num_children_ >> 1];
            if (double_left_half && (1 << right_half_first_child->duplication_factor_) == cur_node->num_children_ >> 1) {
                // pull up right child if all children in the right half are the same
                pull_up_right_child = true;
                left_split = new (AlexModelNode::alloc_type(allocator_).allocate(1)) AlexModelNode(cur_node->level_, allocator_);
            }
            else if (!double_left_half && (1 << left_half_first_child->duplication_factor_) == cur_node->num_children_ >> 1) {
                // pull up left child if all children in the left half are the same
                pull_up_left_child = true;
                right_split = new (AlexModelNode::alloc_type(allocator_).allocate(1)) AlexModelNode(cur_node->level_, allocator_);
            }
            else {
                left_split = new (AlexModelNode::alloc_type(allocator_).allocate(1)) AlexModelNode(cur_node->level_, allocator_);
                right_split = new (AlexModelNode::alloc_type(allocator_).allocate(1)) AlexModelNode(cur_node->level_, allocator_);
            }

            // Do the split
            AlexNode* next_left_split = nullptr;
            AlexNode* next_right_split = nullptr;
            if (double_left_half) {
                // double left half
                assert(left_split != nullptr);
                if (path_idx == static_cast<int>(traversal_path.size()) - 1) {
                    *new_parent = left_split;
                }
                left_split->num_children_ = cur_node->num_children_;
                left_split->children_ = new (AlexModelNode::pointer_alloc_type(allocator_).allocate(left_split->num_children_))
                                             AlexNode*[left_split->num_children_];
                left_split->model_.a_ = cur_node->model_.a_ * 2;
                left_split->model_.b_ = cur_node->model_.b_ * 2;
                int cur = 0;
                while (cur < cur_node->num_children_ >> 1) {
                    AlexNode* cur_child = cur_node->children_[cur];
                    int cur_child_repeats = 1 << cur_child->duplication_factor_;
                    for (int i = 2 * cur; i < 2 * (cur + cur_child_repeats); i++) {
                        left_split->children_[i] = cur_child;
                    }
                    cur_child->duplication_factor_++;
                    cur += cur_child_repeats;
                }
                assert(cur == cur_node->num_children_ >> 1);

                if (pull_up_right_child) {
                    next_right_split = cur_node->children_[cur_node->num_children_ >> 1];
                    next_right_split->level_ = cur_node->level_;
                }
                else {
                    right_split->num_children_ = cur_node->num_children_ >> 1;
                    right_split->children_ = new (AlexModelNode::pointer_alloc_type(allocator_).allocate(right_split->num_children_))
                                                 AlexNode*[right_split->num_children_];
                    right_split->model_.a_ = cur_node->model_.a_;
                    right_split->model_.b_ = cur_node->model_.b_ - cur_node->num_children_ / 2;
                    int j = 0;
                    for (int i = cur_node->num_children_ / 2; i < cur_node->num_children_; i++) {
                        right_split->children_[j] = cur_node->children_[i];
                        j++;
                    }
                    next_right_split = right_split;
                }

                int new_bucketID = path_node.bucketID << 1;
                int repeats = 1 << (prev_left_split->duplication_factor_ + 1);
                int start_bucketID = new_bucketID - (new_bucketID % repeats);  // first bucket with same child
                int mid_bucketID = start_bucketID + (repeats >> 1);
                int end_bucketID = start_bucketID + repeats;  // first bucket with next child
                for (int i = start_bucketID; i < mid_bucketID; i++) {
                    left_split->children_[i] = prev_left_split;
                }
                for (int i = mid_bucketID; i < end_bucketID; i++) {
                    left_split->children_[i] = prev_right_split;
                }
                next_left_split = left_split;
            }
            else {
                // double right half
                assert(right_split != nullptr);
                if (path_idx == static_cast<int>(traversal_path.size()) - 1) {
                    *new_parent = right_split;
                }
                if (pull_up_left_child) {
                    next_left_split = cur_node->children_[0];
                    next_left_split->level_ = cur_node->level_;
                }
                else {
                    left_split->num_children_ = cur_node->num_children_ >> 1;
                    left_split->children_ = new (AlexModelNode::pointer_alloc_type(allocator_).allocate(left_split->num_children_))
                                                 AlexNode*[left_split->num_children_];
                    left_split->model_.a_ = cur_node->model_.a_;
                    left_split->model_.b_ = cur_node->model_.b_;
                    int j = 0;
                    for (int i = 0; i < cur_node->num_children_ >> 1; i++) {
                        left_split->children_[j] = cur_node->children_[i];
                        j++;
                    }
                    next_left_split = left_split;
                }

                right_split->num_children_ = cur_node->num_children_;
                right_split->children_ = new (AlexModelNode::pointer_alloc_type(allocator_).allocate(right_split->num_children_))
                                              AlexNode*[right_split->num_children_];
                right_split->model_.a_ = cur_node->model_.a_ * 2;
                right_split->model_.b_ = (cur_node->model_.b_ - (cur_node->num_children_ >> 1)) * 2;
                int cur = cur_node->num_children_ >> 1;
                while (cur < cur_node->num_children_) {
                    AlexNode* cur_child = cur_node->children_[cur];
                    int cur_child_repeats = 1 << cur_child->duplication_factor_;
                    int right_child_idx = cur - (cur_node->num_children_ >> 1);
                    for (int i = 2 * right_child_idx; i < 2 * (right_child_idx + cur_child_repeats); i++) {
                        right_split->children_[i] = cur_child;
                    }
                    cur_child->duplication_factor_++;
                    cur += cur_child_repeats;
                }
                assert(cur == cur_node->num_children_);

                int new_bucketID = (path_node.bucketID - (cur_node->num_children_ >> 1)) << 1;
                int repeats = 1 << (prev_left_split->duplication_factor_ + 1);
                int start_bucketID = new_bucketID - (new_bucketID % repeats);  // first bucket with same child
                int mid_bucketID = start_bucketID + (repeats >> 1);
                int end_bucketID = start_bucketID + repeats;  // first bucket with next child
                for (int i = start_bucketID; i < mid_bucketID; i++) {
                    right_split->children_[i] = prev_left_split;
                }
                for (int i = mid_bucketID; i < end_bucketID; i++) {
                    right_split->children_[i] = prev_right_split;
                }
                next_right_split = right_split;
            }
            assert(next_left_split != nullptr && next_right_split != nullptr);
            
            to_delete.push_back(cur_node);
            if (!pull_up_left_child && !pull_up_right_child) {
                stats_.num_model_nodes++;
            }
            // This is the expected duplication factor; it will be correct once we split/expand the parent
            next_left_split->duplication_factor_ = cur_node->duplication_factor_;
            next_right_split->duplication_factor_ = cur_node->duplication_factor_;
            prev_left_split = next_left_split;
            prev_right_split = next_right_split;
            path_idx--;
        }

        // Insert into the top node
        const TraversalNode& top_path_node = traversal_path[path_idx];
        AlexModelNode* top_node = top_path_node.node;
        assert(top_node->level_ == stop_propagation_level);
        if (path_idx == static_cast<int>(traversal_path.size()) - 1) {
            *new_parent = top_node;
        }
        int top_bucketID = top_path_node.bucketID;
        int repeats = 1 << prev_left_split->duplication_factor_; // this was the duplication factor of the child that was deleted
        
        // Expand the top node if necessary
        if (repeats == 1) {
            stats_.num_model_node_expansions++;
            stats_.num_model_node_expansion_pointers += top_node->num_children_;
            top_node->expand(1);  // double size of top node
            top_bucketID <<= 1;
            repeats <<= 1;
        }
        else {
            prev_left_split->duplication_factor_--;
            prev_right_split->duplication_factor_--;
        }

        int start_bucketID = top_bucketID - (top_bucketID % repeats);  // first bucket with same child
        int mid_bucketID = start_bucketID + (repeats >> 1);
        int end_bucketID = start_bucketID + repeats;  // first bucket with next child
        for (int i = start_bucketID; i < mid_bucketID; i++) {
            top_node->children_[i] = prev_left_split;
        }
        for (int i = mid_bucketID; i < end_bucketID; i++) {
            top_node->children_[i] = prev_right_split;
        }
        for (auto node : to_delete) {
            delete_node(node);
        }
        return new_data_node;
    }

    /*** Delete ***/
    public:
    // Erases the left-most key with the given key value
    int erase_one(const double& key) {
        AlexDataNode* leaf = get_leaf(key);
        int num_erased = leaf->erase_one(key);
        stats_.num_keys -= num_erased;
        if (leaf->num_keys_ == 0) {
            merge(leaf, key);
        }
        if (key > istats_.key_domain_max_) {
            istats_.num_keys_above_key_domain -= num_erased;
        }
        else if (key < istats_.key_domain_min_) {
            istats_.num_keys_below_key_domain -= num_erased;
        }
        return num_erased;
    }
    // Erases all keys with a certain key value
    int erase(const double& key) {
        AlexDataNode* leaf = get_leaf(key);
        int num_erased = leaf->erase(key);
        stats_.num_keys -= num_erased;
        if (leaf->num_keys_ == 0) {
            merge(leaf, key);
        }
        if (key > istats_.key_domain_max_) {
            istats_.num_keys_above_key_domain -= num_erased;
        }
        else if (key < istats_.key_domain_min_) {
            istats_.num_keys_below_key_domain -= num_erased;
        }
        return num_erased;
    }
    // Erases element pointed to by iterator
    void erase(Iterator it) {
        if (it.is_end()) return;
        double key = it.key();
        it.cur_leaf_->erase_one_at(it.cur_idx_);
        stats_.num_keys--;
        if (it.cur_leaf_->num_keys_ == 0) {
            merge(it.cur_leaf_, key);
        }
        if (key > istats_.key_domain_max_) {
            istats_.num_keys_above_key_domain--;
        }
        else if (key < istats_.key_domain_min_) {
            istats_.num_keys_below_key_domain--;
        }
    }
    // Removes all elements
    void clear() {
        for (NodeIterator node_it = NodeIterator(this); !node_it.is_end(); node_it.next()) {
            delete_node(node_it.current());
        }
        auto empty_data_node = new (AlexDataNode::alloc_type(allocator_).allocate(1)) AlexDataNode(key_less_, allocator_);
        empty_data_node->bulk_load(nullptr, 0);
        root_node_ = empty_data_node;
        create_superroot();
        stats_.num_keys = 0;
    }

    private:
    // Try to merge empty leaf, which can be traversed to by looking up key
    // This may cause the parent node to merge up into its own parent
    void merge(AlexDataNode* leaf, double key) {
        // first save the complete path down to data node
        std::vector<TraversalNode> traversal_path;
        auto leaf_dup = get_leaf(key, &traversal_path);
        // We might need to correct the traversal path in edge cases
        if (leaf_dup != leaf) {
            if (leaf_dup->prev_leaf_ == leaf) {
                correct_traversal_path(leaf, traversal_path, true);
            }
            else if (leaf_dup->next_leaf_ == leaf) {
                correct_traversal_path(leaf, traversal_path, false);
            }
            else {
                assert(false);
                return;
            }
        }
        if (traversal_path.size() == 1) return;

        int path_pos = static_cast<int>(traversal_path.size()) - 1;
        TraversalNode tn = traversal_path[path_pos];
        AlexModelNode* parent = tn.node;
        int bucketID = tn.bucketID;
        int repeats = 1 << leaf->duplication_factor_;

        while (path_pos >= 0) {
            // repeatedly merge leaf with "sibling" leaf by redirecting pointers in the parent
            while (leaf->num_keys_ == 0 && repeats < parent->num_children_) {
                int start_bucketID = bucketID - (bucketID % repeats);
                int end_bucketID = start_bucketID + repeats;
                // determine if the potential sibling leaf is adjacent to the right or left
                bool adjacent_to_right = (bucketID % (repeats << 1) == bucketID % repeats);
                AlexDataNode* adjacent_leaf = nullptr;

                // check if adjacent node is a leaf
                if (adjacent_to_right && parent->children_[end_bucketID]->is_leaf_) {
                    adjacent_leaf = static_cast<AlexDataNode*>(parent->children_[end_bucketID]);
                }
                else if (!adjacent_to_right && parent->children_[start_bucketID - 1]->is_leaf_) {
                    adjacent_leaf = static_cast<AlexDataNode*>(parent->children_[start_bucketID - 1]);
                }
                else {
                    break;  // unable to merge with sibling leaf
                }

                // check if adjacent node is a sibling
                if (leaf->duplication_factor_ != adjacent_leaf->duplication_factor_) {
                    break;  // unable to merge with sibling leaf
                }

                // merge with adjacent leaf
                for (int i = start_bucketID; i < end_bucketID; i++) {
                    parent->children_[i] = adjacent_leaf;
                }
                if (adjacent_to_right) {
                    adjacent_leaf->prev_leaf_ = leaf->prev_leaf_;
                    if (leaf->prev_leaf_) {
                        leaf->prev_leaf_->next_leaf_ = adjacent_leaf;
                    }
                }
                else {
                    adjacent_leaf->next_leaf_ = leaf->next_leaf_;
                    if (leaf->next_leaf_) {
                        leaf->next_leaf_->prev_leaf_ = adjacent_leaf;
                    }
                }
                adjacent_leaf->duplication_factor_++;
                delete_node(leaf);
                stats_.num_data_nodes--;
                leaf = adjacent_leaf;
                repeats = 1 << leaf->duplication_factor_;
            }

            // try to merge up by removing parent and replacing pointers to parent with pointers to leaf in grandparent
            if (repeats == parent->num_children_) {
                leaf->duplication_factor_ = parent->duplication_factor_;
                repeats = 1 << leaf->duplication_factor_;
                bool is_root_node = (parent == root_node_);
                delete_node(parent);
                stats_.num_model_nodes--;

                if (is_root_node) {
                    root_node_ = leaf;
                    update_superroot_pointer();
                    break;
                }

                path_pos--;
                tn = traversal_path[path_pos];
                parent = tn.node;
                bucketID = tn.bucketID;
                int start_bucketID = bucketID - (bucketID % repeats);
                int end_bucketID = start_bucketID + repeats;
                for (int i = start_bucketID; i < end_bucketID; i++) {
                    parent->children_[i] = leaf;
                }
            }
            else {
                break;  // unable to merge up
            }
        }
    }

    /*** Stats ***/

    public:
    // Number of elements
    size_t size() const { return static_cast<size_t>(stats_.num_keys); }

    // True if there are no elements
    bool empty() const { return (size() == 0); }

    // This is just a function required by the STL standard. ALEX can hold more
    // items.
    size_t max_size() const { return size_t(-1); }

    // Size in bytes of all the keys, payloads, and bitmaps stored in this index
    long long data_size() const {
        long long size = 0;
        for (NodeIterator node_it = NodeIterator(this); !node_it.is_end(); node_it.next()) {
            AlexNode* cur = node_it.current();
            if (cur->is_leaf_) {
                size += static_cast<AlexDataNode*>(cur)->data_size();
            }
        }
        return size;
    }

    // Size in bytes of all the model nodes (including pointers) and metadata in
    // data nodes
    long long model_size() const {
        long long size = 0;
        for (NodeIterator node_it = NodeIterator(this); !node_it.is_end(); node_it.next()) {
            size += node_it.current()->node_size();
        }
        return size;
    }

    // Total number of nodes in the RMI
    int num_nodes() const {
        return stats_.num_data_nodes + stats_.num_model_nodes;
    };

    // Number of data nodes in the RMI
    int num_leaves() const { return stats_.num_data_nodes; };

    // Return a const reference to the current statistics
    const struct Stats& get_stats() const { return stats_; }
};

}