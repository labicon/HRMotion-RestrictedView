extends Node2D

var points = [
    Vector2(754.671, 412.418),
    Vector2(344.486, 508.149),
    Vector2(127.251, 240.864),
    Vector2(535.735, 120.297),
    Vector2(754.413, 204.740),
    Vector2(973.505, 319.264),
    Vector2(483.842, 299.161),
    Vector2(752.892, 412.535),
    Vector2(653.537, 124.185),
    Vector2(30.911,  65.518),
    Vector2(498.017, 201.182),
    Vector2(601.642, 118.530),
    Vector2(871.514, 362.297),
    Vector2(538.469, 270.722),
    Vector2(505.249, 478.287),
    Vector2(606.248, 403.393),
    Vector2(367.165, 306.231),
    Vector2(871.000, 303.273),
    Vector2(377.034, 197.638),
    Vector2(452.582, 211.339),
    
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
    score_timer.start()

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
        score -= 10  # Apply a penalty for hitting an obstacle
        update_display()

func update_display():
    # Convert elapsed time to minutes and seconds.
    var minutes = time_elapsed / 60
    var seconds = time_elapsed % 60
    var goal_text = "Goal reached!" if goal_reached else ""
    var label = $GoalLabel  # Adjust the path to your Label node
    label.text = "Score: %d\nTime: %02d:%02d\nObstacles Hit: %d\n%s" % [score, minutes, seconds, obstacles_hit, goal_text]

    # Check if the score is 0 or less and handle the game over logic if necessary.
    if score <= 0 or goal_reached:
        score = 0
        label.text += "\nGame Over"
        score_timer.stop()  # Stop the score timer to stop decreasing score.
        $RestartButton.show()
        $MainMenuButton.show()
        get_tree().paused = true
        # Add any additional game over logic here.


