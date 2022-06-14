ENTRY=scripts/python/tao_build.py
VENV_PATH=/opt/venv/
DEST=/home/fl237079/workspace/tao_built
PPATH=/opt/venv/lib/python3.6/site-packages/disc_dcu/


#yes "" | python ${ENTRY} ${VENV_PATH} -s configure --bridge-gcc default --compiler-gcc default --rocm --rocm_path /opt/rocm-5.1.0
#python ${ENTRY} ${VENV_PATH} -s build_tao_bridge --rocm --rocm_path /opt/rocm-5.1.0
python ${ENTRY} ${VENV_PATH} -s build_tao_compiler --rocm --rocm_path /opt/rocm-5.1.0
#python ${ENTRY} ${VENV_PATH} -s build_mlir_ral --rocm
yes | cp tao/bazel-bin/libtao_ops.so ${DEST}
#yes | cp tao/build/libtao_ops.so  ${DEST}
yes | cp tf_community/bazel-out/k8-opt/bin/tensorflow/compiler/decoupling/tao_compiler_main ${DEST}


yes | cp ${DEST}/libtao_ops.so ${PPATH}
yes | cp ${DEST}/tao_compiler_main ${PPATH}
