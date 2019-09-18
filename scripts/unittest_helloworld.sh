cd unittest
python unittest_network_utils_helloworld.py
python unittest_worker_helloworld.py
cp unittest_master_helloworld.py ../unittest_master_helloworld.py
cd ..
python unittest_master_helloworld.py
rm unittest_master_helloworld.py