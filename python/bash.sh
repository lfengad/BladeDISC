yes | cp /home/fl237079/tao_built/tao_compiler_main disc_dcu/
yes | cp /home/fl237079/tao_built/libtao_ops.so disc_dcu/
rm -rf dist 
python setup.py bdist_wheel
yes | pip uninstall  disc_dcu
pip install dist/*whl
rm -rf build
