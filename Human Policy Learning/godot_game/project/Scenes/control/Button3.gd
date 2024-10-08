extends Button


# Called when the node enters the scene tree for the first time.
func _ready():
    pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
    pass


func _on_pressed():
    var path = ""
    if OS.get_name() == "Windows":
        path = OS.get_environment("APPDATA") + "\\Godot\\app_userdata\\Observation Model Study"
        OS.shell_open(path)
    elif OS.get_name() == "macOS":
        path = OS.get_environment("HOME") + "/Library/Application Support/Godot/app_userdata/Observation Model Study"
        # Use 'open' command directly for macOS, as it handles opening Finder windows well
        OS.execute("open", [path])

    elif OS.get_name() == "X11": # Linux
        path = OS.get_environment("HOME") + "/.local/share/godot/app_userdata/Observation Model Study"
        OS.shell_open(path)
    else:
        print("Unsupported OS")
