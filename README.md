# PI05
1. bash /home/franka/Code/wuzhuangzhe/visual_centric_vla/openpi/serve_policy.sh
   3个任务setting：
   - --policy.config=pi05_pickplace --policy.dir=checkpoints_/pi05_pickplace/my_run/29999
   - --policy.config=pi05_coke --policy.dir=checkpoints_/pi05_coke/my_run/29999
   - --policy.config=pi05_pour --policy.dir=checkpoints_/pi05_pour/my_run/29999

2. policy执行：
   conda activate qing_client
   pickplace：python /home/franka/Code/wuzhuangzhe/visual_centric_vla/codes/apply_policy.py --prompt 'A robot arm with red gripper picking up banana and placing it on a white plate,then picking up carrot and placing it on a white plate'

   coke：python /home/franka/Code/wuzhuangzhe/visual_centric_vla/codes/apply_policy.py --prompt 'Place one coke bottle next to the fixed one, then stack the last coke bottle on top of the two base bottles'

conda activate qing_client
python /home/franka/Code/wuzhuangzhe/visual_centric_vla/codes/apply_policy.py --prompt 'pour cola into the beaker to 1/5 of its volume'
   pour：python /home/franka/Code/wuzhuangzhe/visual_centric_vla/codes/apply_policy.py --prompt 'pour cola into the beaker to 1/5 of its volume'
   pour：python /home/franka/Code/wuzhuangzhe/visual_centric_vla/codes/apply_policy.py --prompt pour cola into the beaker to 1/3 of its volume
   pour：python /home/franka/Code/wuzhuangzhe/visual_centric_vla/codes/apply_policy.py --prompt pour cola into the beaker to 1/2 of its volume

   pour：python /home/franka/Code/wuzhuangzhe/visual_centric_vla/codes/apply_policy.py --prompt 'Write the letters 'y','e','s' on the board in order.'


使用方式
开启“第一次闭合后强制 5 秒”：
python codes/apply_policy.py --lock-gripper-close
关闭该功能（默认行为）：
python codes/apply_policy.py


--fix-euler-at-reset 这个参数写死旋转角

环境：qwen-oft
bash /home/franka/Code/wuzhuangzhe/visual_centric_vla/qwen-oft/scripts_robot/serve_policy.sh

执行：
/home/franka/Code/wuzhuangzhe/visual_centric_vla/codes/apply_policy_qwen_oft.py


/mnt/nas/wuzhuangzhe/qwen-oft/exp_robot