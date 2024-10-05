# Racecar
>Here is a set up guide for onboard development using Jetson Orin Nano. Should also work for most of other jetson boards, but no guarentees.

## [Setting up Git]
- Configure local client
    ~~~bash
    git config --global user.name "your_github_username"
    git config --global user.email "your_github_email"
    git config -l
    ~~~

- Then clone Racecar repo, put in your personal access token in password
    ~~~bash
    git clone https://github.com/armlabstanford/Racecar.git
    > Cloning into `YOUR-REPOSITORY`...
    Username: <type your username>
    Password: <type your password or personal access token (GitHub)
    ~~~
- Go to repo and save your credential
    ~~~bash
    git pull -v
    git config --global credential.helper cache
    ~~~

## [Setting up ssh keyless remote access]
- See if you already have a ssh key generated:
    ~~~bash
    ls -al ~/.ssh/id_*.pub
    ~~~
- If not, generate a key pair:
    ~~~bash
    ssh-keygen -t rsa -b 4096 -C ""
    ~~~
- Throw your id_rsa.pub content into racecar home/.ssh/authorized_keys

## [Setting up ROS Noetic]
- [Refer to this ROS official guide](http://wiki.ros.org/noetic/Installation/Ubuntu)
- Also need ddynamic reconfig
    ~~~bash
    sudo apt install ros-noetic-ddynamic-reconfigure
    ~~~
- for realsense-ros, add
    ~~~c++
    1. find_package( OpenCV REQUIRED )
    2. include_directories(
    include
    ${realsense2_INCLUDE_DIR}
    ${catkin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    )
    3. target_link_libraries(${PROJECT_NAME}
    ${realsense2_LIBRARY}
    ${catkin_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
    ${OpenCV_LIBRARIES}
    )
    ~~~
- You might need to add this line if catkin doesn't find the realsense package
  ~~~c++
  set(realsense2_DIR /usr/lib/x86_64-linux-gnu/cmake/realsense2)
  ~~~

## [Setting up telemetry]
- install pip3 and jtop
    ~~~bash
    sudo apt install python3-pip ros-noetic-rosbridge-server
    sudo pip3 install -U jetson-top
    ~~~
    
## [Setting up mamba env]
- Install mamba
    ~~~bash
    cd ~/Downloads
    wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-pypy3-Linux-aarch64.sh .
    bash Mambaforge-pypy3-Linux-aarch64.sh
    ~~~
- Create environment and install packages
    ~~~bash
    mamba create -n racecar python=3.9
    mamba activate racecar
    mamba install scipy
    pip install pythoncrc numpy evdev pyOpenSSL twisted autobahn service-identity tornado pymongo empy catkin_pkg pyyaml rospkg Pillow defusedxml pycryptodomex gnupg tqdm
    pip install git+https://github.com/LiamBindle/PyVESC
    pip uninstall PyCRC
    pip install pythoncrc
    ~~~
- Don't forget to go to pyvesc to comment the get firmware line and unindent the next line.
- Also need to add your user to the dialout group
    ~~~bash
    sudo usermod -a -G dialout $USER
    ~~~

## [Setting up OpenCV]
- Download opencv-3.16 opencv-contrib-3.16 from github
- Create a folder named opencv/ in home directory, unzip both to folder
- Install necessary packages
    ~~~bash
    sudo apt install build-essential cmake git pkg-config libgtk-3-dev \
          libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
          libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
          gfortran openexr libatlas-base-dev python3-dev python3-numpy \
          libtbb2 libtbb-dev libdc1394-22-dev
    ~~~
- Go to opencv/ execute commmands below:
    ~~~bash
    mkdir -p build && cd build
    cmake -DWITH_CUDA=ON -DENABLE_FAST_MATH=1 -DCUDA_FAST_MATH=1 -DWITH_CUBLAS=1 -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-3.4.16/modules ../opencv-3.4.16/
    cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local  -D INSTALL_C_EXAMPLES=ON  -D INSTALL_PYTHON_EXAMPLES=ON  -D OPENCV_GENERATE_PKGCONFIG=ON  -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib-3.4.16/modules -D BUILD_EXAMPLES=ON ../opencv-3.4.16
    make -j6
    ~~~
- Now go to the cmakelist.txt of racecar/ change OpenCV_DIR to your build path. E.g:
    ~~~bash
    set(OpenCV_DIR /home/racecar/Documents/opencv/build)
    ~~~

## [Setting up librealsense]
- Intel has no official release for arm64 cpu, so we have to build it ourself
- We are referring to [this guide](https://www.lieuzhenghong.com/how_to_install_librealsense_on_the_jetson_nx/) to build the librealsense ourself.
    ~~~bash
    sudo apt-get update && sudo apt-get -y upgrade
    sudo apt-get install -y git libssl-dev libusb-1.0-0-dev pkg-config libgtk-3-dev
    cd ~/Documents
    git clone https://github.com/IntelRealSense/librealsense.git
    cd ./librealsense
    # Remove all realsense cameras before run
    ./scripts/setup_udev_rules.sh
    # Start build
    mkdir build && cd build
    cmake ../ -DBUILD_PYTHON_BINDINGS:bool=true
    sudo make uninstall && sudo make clean && sudo make -j6 && sudo make install
    ~~~

## [TODOs]
- Add launch script
- Add telemetry publishers
- Add additional driving camera
- Add current sensing

## [Setting up video stream]
- On source pc (jetson) You may replace host= with whatever target ip you want to send to
    ~~~bash
    gst-launch-1.0 v4l2src device=/dev/video4 ! 'video/x-raw,width=960,height=540,framerate=60/1' ! videoconvert ! 'video/x-raw,format=I420' ! x264enc speed-preset="ultrafast" tune=zerolatency option-string="sps-id=0" ! rtph264pay ! udpsink host=KWA20.local port=5003 sync=false -e
    ~~~
    gst-launch-1.0 v4l2src device=/dev/video0 ! 'image/jpeg,width=1280,height=720,framerate=30/1' ! jpegdec ! videoconvert ! 'video/x-raw,format=I420' ! x264enc speed-preset="ultrafast" tune="zerolatency" option-string="sps-id=0" ! rtph264pay ! udpsink host=Kens-MacBook-Pro-2.local port=5003 sync=false -e

    gst-launch-1.0 v4l2src device=/dev/video0 ! 'image/jpeg,width=1280,height=720,framerate=30/1' ! jpegdec ! videoconvert ! 'video/x-raw,format=I420' ! x264enc speed-preset="ultrafast" tune="zerolatency" option-string="sps-id=0" ! rtph264pay ! udpsink host=KWA20.local port=5003 sync=false -e

- Wait 30s, and go to target pc (your laptop)
    ~~~bash
    gst-launch-1.0 -v udpsrc port=5003 ! application/x-rtp,encoding-name=H264,payload=96 ! rtph264depay ! avdec_h264 ! videoconvert ! autovideosink
    ~~~

- performance: 
    - rosbridge, uncompressed: 600ms
    - rosbridge, compressed: 160ms
    - gst: 70ms (only 1 viewer restriction applies, although you can relay it to ros)
    - local 30fps: 33ms



- useful tools:
~~~bash
ffplay /dev/video0
v4l2-ctl --device=/dev/video0 --all
~~~

On macos the gst-launch is at: /Library/Frameworks/GStreamer.framework/Commands/gst-launch-1.0