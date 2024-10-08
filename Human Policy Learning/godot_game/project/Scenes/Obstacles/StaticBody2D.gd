extends StaticBody2D

var speed = 500
var pos: Vector2 = Vector2.ZERO

# Called when the node enters the scene tree for the first time.
func _ready():
    pos = Vector2(700, 200)
    position = pos
    


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
    pass
