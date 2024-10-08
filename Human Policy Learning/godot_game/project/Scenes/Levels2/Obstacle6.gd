extends CharacterBody2D

var obstacle_position_data = []  # Array to hold the obstacle's position data
var speed_y = 150
var spedd_x = 0
var pos: Vector2 = Vector2.ZERO
var point = Vector2.ZERO
var direction_x = 1
var direction_y = 1

func _ready():
    add_to_group("obstacles")
    point = get_parent().points[5]
    position = point
    
func _process(delta):
    var prob = get_parent().probability
    position.y += speed_y * direction_y * delta
    position.x += spedd_x * direction_x * delta
    
    if position.y > 640:
        position.y = 640
        direction_y = -1
    elif position.y < 0:
        position.y = 0
        direction_y = 1
    
    if position.x > 1150:
        position.x = 1150
        direction_x = -1
    elif position.x < 0:
        position.x = 0
        direction_x = 1
        
    if position.y == 250 and direction_y == 1 and randf() <= prob:
        # Stop the obstacle
        set_process(false)  # or obstacle.set_physics_process(false), depending on your needs

        # Wait for 2 seconds
        await get_tree().create_timer(2.0).timeout

        # Resume the obstacle
        set_process(true)  # or obstacle.set_physics_process(true)
        
    if position.y == 450 and direction_y == -1 and randf() <= prob:
        # Stop the obstacle
        set_process(false)  # or obstacle.set_physics_process(false), depending on your needs

        # Wait for 2 seconds
        await get_tree().create_timer(2.0).timeout

        # Resume the obstacle
        set_process(true)  # or obstacle.set_physics_process(true)
        
    obstacle_position_data.append({"x": position.x, "y": position.y})

func save_obstacle_trajectory(light_value, game_mode, game_selected, time, prob):
    # Check if game_mode is 0, and if so, do not save the data
    if game_mode == 0:
        print("Game mode is 0, not saving obstacle trajectory data.")
        return  # Exit the function early
        
    print("save_obstacle_trajectory function called")  # Debug print

    # Format the subfolder structure
    var base_path = "user://"
    var folder_path = "{0}/{1}/{2}/{3}/".format([game_selected, light_value, prob, time], "{_}")
    var full_path = base_path + folder_path

    # Ensure the directory exists
    var dir = DirAccess.open(base_path)
    if not dir.dir_exists(full_path):
        dir.make_dir_recursive(full_path)

    # Specify the file name and path
    var file_name = "obstacle6_trajectory_data.json"
    var path = full_path + file_name

    # Open the file for writing
    var file = FileAccess.open(path, FileAccess.WRITE)
    var json = JSON.new()
    var json_string = json.stringify(obstacle_position_data)  # Ensure obstacle_position_data is defined somewhere
    file.store_string(json_string)
    file.close()
    
    print("Obstacle trajectory data saved to ", path)
