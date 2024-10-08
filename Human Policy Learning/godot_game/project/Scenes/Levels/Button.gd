extends Button

# Called when the node enters the scene tree for the first time.
func _ready():
    pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
    pass

func _on_pressed():
    get_tree().paused = false
    print("restart")
    var levels_node = get_parent()
    levels_node.trials += 1 
    get_tree().change_scene_to_file("res://Scenes/control/control.tscn")


func _on_button_start_pressed():
    pass # Replace with function body.
