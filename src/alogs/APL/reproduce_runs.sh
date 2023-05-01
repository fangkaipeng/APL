# DomainNet
python3 main.py -data DomainNet -hd sketch -sd quickdraw -bs 50 -log 15 -lr 0.0001 -use_NTP 0 -tp_N_CTX 16 -GP_CLS_NUM_TOKENS 1 -GP_DOM_NUM_TOKENS 1 -debug_mode 0 
python3 main.py -data DomainNet -hd quickdraw -sd sketch -bs 50 -log 15 -lr 0.0001 -use_NTP 0 -tp_N_CTX 1 -GP_CLS_NUM_TOKENS 1 -GP_DOM_NUM_TOKENS 1 -debug_mode 0 
python3 main.py -data DomainNet -hd clipart -sd painting -bs 50 -log 15 -lr 0.0001 -use_NTP 0 -tp_N_CTX 16 -GP_CLS_NUM_TOKENS 1 -GP_DOM_NUM_TOKENS 1 -debug_mode 0 
python3 main.py -data DomainNet -hd painting -sd infograph -bs 50 -log 15 -lr 0.0001 -use_NTP 0 -tp_N_CTX 16 -GP_CLS_NUM_TOKENS 1 -GP_DOM_NUM_TOKENS 1 -debug_mode 0 
python3 main.py -data DomainNet -hd infograph -sd painting -bs 50 -log 15 -lr 0.0001 -use_NTP 0 -tp_N_CTX 16 -GP_CLS_NUM_TOKENS 1 -GP_DOM_NUM_TOKENS 1 -debug_mode 0 

# Sketchy
python3 main.py -data Sketchy -bs 50 -log 15 -lr 0.0001 -use_NTP 0 -tp_N_CTX 16 -GP_CLS_NUM_TOKENS 1 -GP_DOM_NUM_TOKENS 1 -debug_mode 0 

# TUBerlin
python3 main.py -data TUBerlin -bs 50 -log 15 -lr 0.0001 -use_NTP 0 -tp_N_CTX 16 -GP_CLS_NUM_TOKENS 1 -GP_DOM_NUM_TOKENS 1 -debug_mode 0 
