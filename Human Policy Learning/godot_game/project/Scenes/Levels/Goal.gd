extends CharacterBody2D


var speed = 500
var pos: Vector2 = Vector2.ZERO
var point = Vector2(1050, 620)

func _ready():
    position = point

