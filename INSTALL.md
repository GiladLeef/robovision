# Installing Python 3.10.9 and Running Robovision on Raspberry Pi

This guide provides step-by-step instructions for installing Python 3.10.9 on a Raspberry Pi and running Robovision.

## Prerequisites

- Raspberry Pi with Raspbian OS (or compatible)
- Internet connection

## Installing Python 3.10.9

1. **Update Package Lists:**

    ```bash
    sudo apt-get update
    ```

2. **Install Dependencies:**

    ```bash
    sudo apt-get install -y build-essential zlib1g-dev libffi-dev libssl-dev libncurses5-dev libsqlite3-dev libreadline-dev libbz2-dev liblzma-dev libgdbm-dev tk-dev libdb-dev libpcap-dev libgl1-mesa-glx
    ```

3. **Download Python 3.10.9 Source Code:**

    ```bash
    wget https://www.python.org/ftp/python/3.10.9/Python-3.10.9.tgz
    ```

4. **Extract the Archive:**

    ```bash
    tar -xvf Python-3.10.9.tgz
    ```

5. **Navigate to the Python Directory:**

    ```bash
    cd Python-3.10.9
    ```

6. **Configure and Install Python:**

    ```bash
    ./configure
    make -j4
    sudo make altinstall
    ```

   Note: Using `make altinstall` avoids replacing the default system Python.

7. **Check Python Version:**

    ```bash
    python3.10 --version
    ```

   This should display `Python 3.10.9`.

## Running Robovision

1. **Navigate to Robovision Directory:**

    ```bash
    cd /opt/robovision
    ```

2. **Create a Virtual Environment:**

    ```bash
    python3.10 -m venv venv
    ```

3. **Activate the Virtual Environment:**

    ```bash
    source venv/bin/activate
    ```

4. **Install Dependencies:**

    ```bash
    pip install numpy onnxruntime opencv-python pynetworktables
    ```

5. **Run Robovision:**

    ```bash
    sudo su && cd /opt/robovision && source venv/bin/activate && python3.10 webcamntv.py
    ```

   This command switches to the superuser (`sudo su`), navigates to the Robovision directory, activates the virtual environment, and runs the `webcamntv.py` script.

## Additional Notes

- Make sure to adjust the number in the `-j` option in the `make` command based on the number of cores in your Raspberry Pi for faster compilation.

- If you encounter any issues or want more information, refer to the [official Python documentation](https://docs.python.org/3/using/unix.html) and [Robovision documentation](https://robovision.ai/docs).

---  
