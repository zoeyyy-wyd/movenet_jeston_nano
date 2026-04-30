# # 跌倒视频的 annotation
# cat ~/movenet_jeston_nano/data_Le2i/"Coffee_room_01/Coffee_room_01/Annotation_files/video (1).txt"

# 非跌倒视频的 annotation (你说有 0,0 的, 找一个看)
# 先看哪些是 0,0 (前两个数都是 0)
for f in ~/movenet_jeston_nano/data_Le2i/"Coffee_room_01/Coffee_room_01/Annotation_files/"*.txt; do
    head -2 "$f" | tr '\n' ' '
    echo " <- $(basename "$f")"
done | head -20