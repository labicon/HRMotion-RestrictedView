extends Control

@onready var button1 = $light1
@onready var button2 = $light2
@onready var button3 = $light3
@onready var howtoplay = $howtoplay
@onready var howtorecord = $howtorecord
@onready var howtoplay2 = $howtoplay2
func _ready():
    # Connect the 'toggled' signal for each toggleable button
    pass

func _on_play_button_pressed():
    howtoplay.visible = true
    howtorecord.visible = false
    howtoplay2.visible = false

func _on_record_button_pressed():
    howtoplay.visible = false
    howtorecord.visible = true
    howtoplay2.visible = false


func _on_play_button_2_pressed():
    howtoplay.visible = false
    howtorecord.visible = false
    howtoplay2.visible = true
