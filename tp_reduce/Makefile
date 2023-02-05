build:
	cmake -B build

compile: build
	make -C build -j8

run: compile
	./build/bench
