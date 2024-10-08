extends CharacterBody2D

var obstacle_position_data = []  # Array to hold the obstacle's position data
var speed = 300
var pos: Vector2 = Vector2.ZERO
var point = Vector2.ZERO

func _ready():
    add_to_group("obstacles")
    point = get_parent().points[10]
    position = point
    # Record current position
    obstacle_position_data.append({"x": position.x, "y": position.y})

func save_obstacle_trajectory(light_value, game_mode, game_selected, time):
    # Check if game_mode is 0, and if so, do not save the data
    if game_mode == 0:
        print("Game mode is 0, not saving obstacle trajectory data.")
        return  # Exit the function early
        
    print("save_obstacle_trajectory function called")  # Debug print

    # Format the subfolder structure
    var base_path = "user://"
    var folder_path = "{0}/{1}/{2}/".format([game_selected, light_value, time], "{_}")
    var full_path = base_path + folder_path

    # Ensure the directory exists
    var dir = DirAccess.open(base_path)
    if not dir.dir_exists(full_path):
        dir.make_dir_recursive(full_path)

    # Specify the file name and path
    var file_name = "obstacle11_trajectory_data.json"
    var path = full_path + file_name

    # Open the file for writing
    var file = FileAccess.open(path, FileAccess.WRITE)
    var json = JSON.new()
    var json_string = json.stringify(obstacle_position_data)  # Ensure obstacle_position_data is defined somewhere
    file.store_string(json_string)
    file.close()
    
    print("Obstacle trajectory data saved to ", path)
