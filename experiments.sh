# !/bin/sh
python main.py ridge WESAD 25  mrt 0.2
python main.py ridge WESAD 50  mrt 0.2
python main.py ridge WESAD 75  mrt 0.2
python main.py ridge WESAD 100 mrt 0.2
python main.py ridge HHAR  25  mrt 0.2
python main.py ridge HHAR  50  mrt 0.2
python main.py ridge HHAR  75  mrt 0.2
python main.py ridge HHAR  100 mrt 0.2
#
python main.py incfed WESAD 25  mrt 1
python main.py incfed WESAD 50  mrt 1
python main.py incfed WESAD 75  mrt 2
python main.py incfed WESAD 100 mrt 2
python main.py incfed HHAR  25  mrt 1
python main.py incfed HHAR  50  mrt 1
python main.py incfed HHAR  75  mrt 1
python main.py incfed HHAR  100 mrt 2
#
python main.py ip WESAD 25  mrt 0.2
python main.py ip WESAD 50  mrt 0.2
python main.py ip WESAD 75  mrt 0.2
python main.py ip WESAD 100 mrt 0.2
python main.py ip HHAR  25  mrt 0.2
python main.py ip HHAR  50  mrt 0.2
python main.py ip HHAR  75  mrt 0.2
python main.py ip HHAR  100 mrt 0.2
#
python main.py fedip WESAD 25  mrt 1
python main.py fedip WESAD 50  mrt 1
python main.py fedip WESAD 75  mrt 1
python main.py fedip WESAD 100 mrt 2
python main.py fedip HHAR  25  mrt 1
python main.py fedip HHAR  50  mrt 1
python main.py fedip HHAR  75  mrt 1
python main.py fedip HHAR  100 mrt 2
#
python main.py continual_ip_naive WESAD 25  t 0.2
python main.py continual_ip_naive WESAD 50  t 0.2
python main.py continual_ip_naive WESAD 75  t 0.2
python main.py continual_ip_naive WESAD 100 t 0.2
python main.py continual_ip_naive HHAR  25  t 0.2
python main.py continual_ip_naive HHAR  50  t 0.2
python main.py continual_ip_naive HHAR  75  t 0.2
python main.py continual_ip_naive HHAR  100 t 0.2
#
python main.py continual_fedip_naive WESAD 25  t 1
python main.py continual_fedip_naive WESAD 50  t 2
python main.py continual_fedip_naive WESAD 75  t 2
python main.py continual_fedip_naive WESAD 100 t 2
python main.py continual_fedip_naive HHAR  25  t 1
python main.py continual_fedip_naive HHAR  50  t 2
python main.py continual_fedip_naive HHAR  75  t 2
python main.py continual_fedip_naive HHAR  100 t 2
#
python main.py continual_ip_replay WESAD 25  t 0.2
python main.py continual_ip_replay WESAD 50  t 0.2
python main.py continual_ip_replay WESAD 75  t 0.2
python main.py continual_ip_replay WESAD 100 t 0.2
python main.py continual_ip_replay HHAR  25  t 0.2
python main.py continual_ip_replay HHAR  50  t 0.2
python main.py continual_ip_replay HHAR  75  t 0.2
python main.py continual_ip_replay HHAR  100 t 0.2
#
python main.py continual_fedip_replay WESAD 25  mrt 1
python main.py continual_fedip_replay WESAD 50  mrt 1
python main.py continual_fedip_replay WESAD 75  mrt 2
python main.py continual_fedip_replay WESAD 100 mrt 2
python main.py continual_fedip_replay HHAR  25  mrt 1
python main.py continual_fedip_replay HHAR  50  mrt 1
python main.py continual_fedip_replay HHAR  75  mrt 1
python main.py continual_fedip_replay HHAR  100 mrt 2
#
python main.py continual_ip_joint WESAD 25  rt 0.2
python main.py continual_ip_joint WESAD 50  rt 0.2
python main.py continual_ip_joint WESAD 75  rt 0.2
python main.py continual_ip_joint WESAD 100 rt 0.2
python main.py continual_ip_joint HHAR  25  rt 0.2
python main.py continual_ip_joint HHAR  50  rt 0.2
python main.py continual_ip_joint HHAR  75  rt 0.2
python main.py continual_ip_joint HHAR  100 rt 0.2
#
python main.py continual_fedip_joint WESAD 25  rt 1
python main.py continual_fedip_joint WESAD 50  rt 1
python main.py continual_fedip_joint WESAD 75  rt 2
python main.py continual_fedip_joint WESAD 100 rt 2
python main.py continual_fedip_joint HHAR  25  rt 1
python main.py continual_fedip_joint HHAR  50  rt 1
python main.py continual_fedip_joint HHAR  75  rt 1
python main.py continual_fedip_joint HHAR  100 rt 2