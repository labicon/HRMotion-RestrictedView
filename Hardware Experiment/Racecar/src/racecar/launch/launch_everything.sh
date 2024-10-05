# Open a new GNOME terminal tab and run commands
run_in_tab() {
    local commands="$1"
    gnome-terminal --tab --geometry=79x20+800+400 -- /bin/bash -c "$commands; exec bash"
}


# roslaunch racecar driving_node_t265.launch

run_in_tab "roslaunch racecar driving_node_t265.launch"

# Start streaming video feed
# run_in_tab "gst-launch-1.0 v4l2src device=/dev/v4l/by-id/usb-Arducam_Technology_Co.__Ltd._Arducam_12MP_SN0001-video-index0 ! 'image/jpeg,width=1280,height=720,framerate=30/1' ! jpegdec ! videoconvert ! 'video/x-raw,format=I420' ! x264enc speed-preset='ultrafast' tune='zerolatency' option-string='sps-id=0' ! rtph264pay ! udpsink host=Kens-MacBook-Pro-2.local port=5003 sync=false -e"

run_in_tab "gst-launch-1.0 v4l2src device=/dev/v4l/by-id/usb-Arducam_Technology_Co.__Ltd._Arducam_12MP_SN0001-video-index0 ! 'image/jpeg,width=1280,height=720,framerate=30/1' ! jpegdec ! videoconvert ! 'video/x-raw,format=I420' ! x264enc speed-preset='ultrafast' tune='zerolatency' option-string='sps-id=0' ! rtph264pay ! udpsink host=KWA20.local port=5003 sync=false -e"