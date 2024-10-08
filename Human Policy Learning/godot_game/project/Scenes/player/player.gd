extends CharacterBody2D

var playerScene = preload("res://Scenes/player/player1.tscn")
var speed = 100
var pos: Vector2 = Vector2.ZERO
var position_data = []  # Array to hold the player's position data

enum FlashlightState {
    r192,
    r96,
    r480
}

var current_flashlight_state = FlashlightState.r480

func _ready():
    pos = Vector2(20, 20)
    position = pos
    
func _process(delta):
    var direction = Input.get_vector("left", "right", "up", "down")
    velocity = direction * speed
    look_at(get_global_mouse_position())
    move_and_slide()
    
    var current_position = global_position
    var theta = rotation_degrees  # Assuming 'rotation' property of this node or a child node represents the flashlight's orientation
    position_data.append({
        "x": current_position.x,
        "y": current_position.y,
        "theta": theta  # Added orientation in degrees
    })
    print(position_data[-1])

func save_trajectory(light_value, game_mode, game_selected, time):
    if game_mode == 0:
        print("Game mode is 0, not saving trajectory data.")
        return  # Exit the function early
        
    print("save_trajectory function called")

    var base_path = "user://"
    var folder_path = "{0}/{1}/{2}/".format([game_selected, light_value, time], "{_}")
    var full_path = base_path + folder_path

    # Ensure the directory exists
    var dir = DirAccess.open(base_path)
    if not dir.dir_exists(full_path):
        dir.make_dir_recursive(full_path)

    var file_name = "trajectory_data.json"
    var path = full_path + file_name
    var file = FileAccess.open(path, FileAccess.WRITE)

    var json = JSON.new()
    var json_string = json.stringify(position_data)
    file.store_string(json_string)
    file.close()
    print("Trajectory data saved to ", path)

func save_all_trajectories(light_value, game_mode, game_selected, time):
    save_trajectory(light_value, game_mode, game_selected, time)  # Save player's trajectory
    var obstacles = get_tree().get_nodes_in_group("obstacles")
    for obstacle in obstacles:
        obstacle.save_obstacle_trajectory(light_value, game_mode, game_selected, time)  # Save each obstacle's trajectory
        
func _input(event):
#    if event.is_action_pressed("save_data"):
#        save_all_trajectories()
        
    if event.is_action_pressed('toggle_flashlight'):
        # Cycle through the flashlight states
        current_flashlight_state += 1
        if current_flashlight_state > FlashlightState.r480:
            current_flashlight_state = FlashlightState.r192

        toggle_flashlights(current_flashlight_state)

func toggle_flashlights(state):
    # Turn all flashlights off
    $r192.visible = false
    $r96.visible = false
    $r480.visible = false
    
    # Turn the selected flashlight on
    match state:
        FlashlightState.r192:
            $r192.visible = true
        FlashlightState.r96:
            $r96.visible = true
        FlashlightState.r480:
            $r480.visible = true
        _:
            pass  # All flashlights stay off
