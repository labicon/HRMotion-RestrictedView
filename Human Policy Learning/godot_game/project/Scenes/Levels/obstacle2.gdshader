shader_type canvas_item;

void fragment() {
    COLOR = texture(TEXTURE, UV);
    COLOR.rgb *= LIGHT; // This multiplies the sprite's color by the light's color
}