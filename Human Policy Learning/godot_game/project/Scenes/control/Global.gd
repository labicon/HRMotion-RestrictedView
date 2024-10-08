# In Global.gd

extends Control

func make_r192_visible():
    var player_scene = get_tree().current_scene
    var r192_node = player_scene.find_node("r192", true, false)
    if r192_node:
        r192_node.visible = true
