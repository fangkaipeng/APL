# DomainNet
python3 main.py -data DomainNet -hd sketch -sd quickdraw  -bs 256  -es 2 -lr 0.001  -log 15 -e 100 -ts TP+VP 
python3 main.py -data DomainNet -hd quickdraw -sd sketch  -bs 256  -es 2 -lr 0.001  -log 15 -e 100 -ts TP+VP 
python3 main.py -data DomainNet -hd painting -sd infograph  -bs 256  -es 2 -lr 0.001  -log 15 -e 100 -ts TP+VP
python3 main.py -data DomainNet -hd infograph -sd painting  -bs 256  -es 2 -lr 0.001  -log 15 -e 100 -ts TP+VP
python3 main.py -data DomainNet -hd clipart -sd painting  -bs 256  -es 2 -lr 0.001  -log 15 -e 100 -ts TP+VP

# Sketchy
python3 main.py -data Sketchy -bs 256  -es 2 -lr 0.001  -log 15 -e 100 -ts TP+VP -debug_mode 0

# TU-Berlin
python3 main.py -data TUBerlin -bs 256  -es 2 -lr 0.001  -log 15 -e 100 -ts TP+VP -debug_mode 0

