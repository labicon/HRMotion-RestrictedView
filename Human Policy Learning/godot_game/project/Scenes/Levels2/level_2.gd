extends Node2D
@onready var light480 = get_node("player/r480")
@onready var light192 = get_node("player/r192")
@onready var light96 = get_node("player/r96")
@onready var player = get_node("player")
@onready var pro_6 = get_node("player/P_60")
@onready var pro_8 = get_node("player/P_80")

var PlayerScene = preload("res://Scenes/player2/player2.tscn")
var game_mode = 0
var game_selected = 2
var time = Time.get_datetime_string_from_system()
var light_value = 0
var trials = 0
var randcode = randi() % 100 + 1
var probability = 0.8

var points = [
    Vector2(754.671, 412.418),
    Vector2(344.486, 508.149),
    Vector2(180.251, 240.864),
    Vector2(535.735, 120.297),
    Vector2(754.413, 204.740),
    Vector2(973.505, 319.264),
    Vector2(483.842, 299.161),
    Vector2(752.892, 412.535),
    Vector2(653.537, 124.185),
    Vector2(160,  65.518),
    Vector2(498.017, 201.182),
    Vector2(601.642, 118.530),
    Vector2(871.514, 362.297),
    Vector2(538.469, 270.722),
    Vector2(505.249, 478.287),
    Vector2(606.248, 403.393),
    Vector2(367.165, 306.231),
    Vector2(871.000, 303.273),
    Vector2(377.034, 197.638),
    Vector2(250, 250),
    Vector2(250, 450),
    ]
# Declare variables to track score, time, and gameplay events.
var score = 100  # Starting score
var score_timer = Timer.new()  # Timer to decrease score and track time
var time_elapsed = 0  # Time since the start of the game in seconds
var obstacles_hit = 0  # Number of obstacles the player has hit
var goal_reached = false  # Whether the goal has been reached

func _ready():
    # Initialize the Label node with the starting score and other information.
    update_display()

    # Set up the timer to decrease score every second and to track elapsed time.
    score_timer.wait_time = 1.0  # Tick every second
    score_timer.autostart = true
    score_timer.timeout.connect(_on_score_timer_timeout)
    add_child(score_timer)
    
    time = time.replace(":", "")
    print(time)
    
    score_timer.start()
    get_tree().paused = true
    
func _on_score_timer_timeout():
    # Decrease score over time and track time elapsed.
    score -= 1
    time_elapsed += 1
    update_display()

func _on_Goal_2d_body_entered(body):
    if body == $player:
        goal_reached = true
        print("Player entered the goal area!")
        update_display()
        score_timer.stop()
        
func _on_area_2d_body_entered(body):
    if body == $player:
        obstacles_hit += 1
        print("Player hit an obstacle!")
        score -= 30  # Apply a penalty for hitting an obstacle
        update_display()

func update_display():
    # Convert elapsed time to minutes and seconds.
    var minutes = time_elapsed / 60
    var seconds = time_elapsed % 60
    var goal_text = "Goal reached!" if goal_reached else ""
    var label = $GoalLabel  # Adjust the path to your Label node
    var pro = probability
    print(probability)
    label.text = "Score: %d\nTime: %02d:%02d\nObstacles Hit: %d\nProbability : %.2f\n%s" % [score, minutes, seconds, obstacles_hit, pro, goal_text]
    # Check if the score is 0 or less and handle the game over logic if necessary.
    if score <= 0 or goal_reached:
        score = 0
        label.text += "\nGame Over"
        score_timer.stop()  # Stop the score timer to stop decreasing score.
        $RestartButton.show()
        $player.save_all_trajectories(light_value, game_mode, game_selected, time, probability)
        get_tree().paused = true
        # Add any additional game over logic here.




func _on_r_192_pressed():
    print(light192)
    light192.visible = true
    light96.visible = false
    light480.visible = false
    light_value = 192
    
func _on_r_96_pressed():
    print(light96)
    light192.visible = false
    light96.visible = true
    light480.visible = false
    light_value = 96
    
func _on_r_480_pressed():
    print(light480)
    light192.visible = false
    light96.visible = false
    light480.visible = true
    light_value = 480



func _on_button_start_pressed():
    pass # Replace with function body.


func _on_start_practice_pressed():
    pass # Replace with function body.


func _on_p_60_pressed():
    probability = 0.4
    print(probability) # Replace with function body.

func _on_r_481_pressed():
    probability = 0.8 # Replace with function body.
    print(probability)
