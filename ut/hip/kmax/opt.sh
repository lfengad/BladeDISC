#/opt/rocm/llvm/bin/opt  -targetlibinfo -tti -tbaa -scoped-noalias -assumption-cache-tracker -profile-summary-info -forceattrs -inferattrs -ipsccp -called-value-propagation -globalopt -domtree -mem2reg -deadargelim -basicaa -aa -loops -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -instcombine -simplifycfg -basiccg -globals-aa -prune-eh -always-inline -functionattrs -sroa -memoryssa -early-cse-memssa -speculative-execution -lazy-value-info -jump-threading -correlated-propagation -libcalls-shrinkwrap -branch-prob -block-freq -pgo-memop-opt -tailcallelim -reassociate -loop-simplify -lcssa-verification -lcssa -scalar-evolution -loop-rotate -licm -loop-unswitch -indvars -loop-idiom -loop-deletion -loop-unroll -memdep -memcpyopt -sccp -demanded-bits -bdce -dse -postdomtree -adce -barrier -rpo-functionattrs -globaldce -float2int -loop-accesses -loop-distribute -loop-vectorize -loop-load-elim -alignment-from-assumptions -strip-dead-prototypes -loop-sink -instsimplify -div-rem-pairs -verify -ee-instrument -early-cse -lower-expect -o $2 $1
/opt/rocm/llvm/bin/opt -O3 -debug-pass-manager -o $2 $1