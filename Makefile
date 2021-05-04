all:
	g++ main.cpp -o main -msse4.1 -mpopcnt
clean:
	rm main