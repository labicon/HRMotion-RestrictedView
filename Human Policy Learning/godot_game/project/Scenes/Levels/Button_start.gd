extends Button

# Called when the node enters the scene tree for the first time.
func _ready():
    pass

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
    pass


func _on_pressed():
    var r96 = get_parent().get_node("r96")
    var r480 = get_parent().get_node("r480")
    var r192 = get_parent().get_node("r192")
    var practice = get_parent().get_node("start_practice")
    get_tree().paused = false
    print('pressed')
    var levels_node = get_parent()
    levels_node.trials += 1
    if levels_node and levels_node.has_method("set_game_mode"):  # Checking if the method exists
        levels_node.set_game_mode(1)  # Assuming a setter method exists
    elif levels_node:
        levels_node.game_mode = 1  # Direct property access
    hide()
    r96.hide()
    r480.hide()
    r192.hide()
    practice.hide()
