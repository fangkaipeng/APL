# DomainNet

python3 main.py -hd sketch -sd quickdraw -bs 512 -es 5 -lr 0.001 -clip_bb ViT-B/32 -log 15 -e 100 -ts LP -debug_mode 0
python3 main.py -hd quickdraw -sd sketch -bs 512 -es 5 -lr 0.001 -clip_bb ViT-B/32 -log 15 -e 100 -ts LP -debug_mode 0
python3 main.py -hd clipart -sd painting -bs 512 -es 5 -lr 0.001 -clip_bb ViT-B/32 -log 15 -e 100 -ts LP -debug_mode 0
python3 main.py -hd painting -sd infograph -bs 512 -es 5 -lr 0.001 -clip_bb ViT-B/32 -log 15 -e 100 -ts LP -debug_mode 0
python3 main.py -hd infograph -sd painting -bs 512 -es 10 -lr 0.001 -clip_bb ViT-B/32 -log 15 -e 100 -ts LP -debug_mode 0

# Sketchy
python3 main.py -data Sketchy -bs 480 -es 3 -lr 0.001 -clip_bb ViT-B/32 -log 15 -e 100 -ts LP -debug_mode 0

# TUBerlin
python3 main.py -data TUBerlin -bs 480 -es 3 -lr 0.001 -clip_bb ViT-B/32 -log 15 -e 100 -ts LP -debug_mode 0
