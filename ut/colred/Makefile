all: run


compile:
	hipcc -o exec ../kernels.cu -I/opt/rocm/include -O3 -DLT=32 -DLW=16 -DLI=8 -DD0=32 -DD1=32 -DD2=64 -DTT=double
	#hipcc -o launch module_launch.cc --std=c++11
	#hipcc kernel.cu -o kernel.hsaco --genco
	#hipcc kernel.cu -o kernel.ll -emit-llvm -S

run: compile
	./exec
