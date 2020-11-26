## Getting started
1. Start the TCP local server. Run
    ```console
    $ python xsens_py_interface/multithread_server.py
    ```
2. Setup and start xsens MTw device. Build and run ./xsens_py_interface/cpp_xsapi_back/Monitor/awindamonitor_cpp/main.cpp (you may need to install QT creator)
3. Follow the instructions on the GUI
4. Run the main sign detect interface. Run
    ```console
    $ python main.py
    ```