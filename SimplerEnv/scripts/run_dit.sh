model_name=dit_policy
tasks=(
  bridge.sh
  # drawer_variant_agg.sh
  # drawer_visual_matching.sh


  # pick_coke_can_variant_agg.sh
  # pick_coke_can_visual_matching.sh
  # put_in_drawer_variant_agg.sh
  # put_in_drawer_visual_matching.sh


  # pick_coke_can_variant_agg_b.sh
  # pick_coke_can_variant_agg_backgrounds.sh
  # pick_coke_can_variant_agg_texture.sh
  # pick_coke_can_variant_agg_distractors.sh
  # pick_coke_can_variant_agg_lightings.sh  
  #   move_near_variant_agg.sh
  # move_near_visual_matching.sh
)

ckpts=(
  # /mnt/petrelfs/houzhi/Code/embodied_foundation/outputs/openx/openx_full_train_o2_p32_wotimestep_resized7_noclamp_filter/2024-08-19/17-47-38/ckpt_102000.pth
  # /mnt/petrelfs/houzhi/Code/embodied_foundation/outputs/openx/openx_full_train_o2_p32_wotimestep_resized7_noclamp_filter/2024-08-19/17-47-38/checkpoints/ckpt_272000.pth
  # /mnt/petrelfs/houzhi/Code/embodied_foundation/outputs/openx/openx_full_train_o2_p32_wotimestep_resized7_noclamp_filter/2024-08-19/17-47-38/ckpt_68000.pth
  # /mnt/petrelfs/houzhi/Code/embodied_foundation/outputs/openx/openx_full_train_o2_p32_wotimestep_resized7_noclamp_filter/2024-08-19/17-47-38/ckpt_192000.pth
  /mnt/petrelfs/houzhi/Code/embodied_foundation/outputs/openx/openx_full_train_o2_p15_aug_sche1/2024-10-21/12-09-46/checkpoints/ckpt_107000.pth
  # /mnt/petrelfs/houzhi/Code/embodied_foundation/oxe_aug_ckpt_111000.pth
  # /mnt/petrelfs/houzhi/Code/temp_uni/outputs/2025-01-20/04-12-39/checkpoints/ckpt_103000.pth
  # /mnt/petrelfs/houzhi/Code/embodied_foundation/sche1_ckpt_84000.pth
  # /mnt/petrelfs/houzhi/Code/embodied_foundation/SimplerEnv-OpenVLA/ckpt_130000.pth
  # /mnt/petrelfs/houzhi/Code/embodied_foundation/outputs/openx/openx_full_train_o2_p50_wotimestep_resized7/2024-09-17/22-23-21/checkpoints/ckpt_200000.pth
  # /mnt/petrelfs/houzhi/Code/embodied_foundation/outputs/openx/openx_full_train_o2_p32_wotimestep_resized71_noclamp_filter/2024-10-01/14-55-12/checkpoints/ckpt_137000.pth
  # /mnt/petrelfs/houzhi/Code/embodied_foundation/outputs/openx/openx_full_train_o2_p31_wotimestep_resized7/2024-09-20/20-57-23/checkpoints/ckpt_111000.pth
  # /mnt/petrelfs/houzhi/Code/embodied_foundation/outputs/openx/openx_full_train_o2_p8_wotimestep_resized71_noclamp_filter/2024-10-11/21-53-31/checkpoints/ckpt_98000.pth
  # /mnt/petrelfs/houzhi/Code/temp_uni/outputs/2025-01-11/05-26-40/checkpoints/ckpt_93000.pth
)

action_ensemble_temp=-0.8
for ckpt_path in ${ckpts[@]}; do
  base_dir=$(dirname $ckpt_path)

  # evaluation in simulator
  # logging_dir=$base_dir/simpler_env/$(basename $ckpt_path)${action_ensemble_temp}
  logging_dir=results_size_d50/$(basename $ckpt_path)${action_ensemble_temp}
  # logging_dir='results_f/ckpt_200000.pth-0.8'
  # logging_dir='results_size/ckpt_107000.pth-0.8'
  # logging_dir=/mnt/petrelfs/houzhi/Code/embodied_foundation/SimplerEnv-OpenVLA/results_size_d20_1/bridge_only_ckpt-0.8
  # mkdir -p $logging_dir
  # for i in ${!tasks[@]}; do
  #   task=${tasks[$i]}
  #   echo "🚀 running $task ..."
  #   device=0
  #   session_name=CUDA${device}-$(basename $logging_dir)-${task}
  #   ORIGSIZE=1 bash scripts/$task $ckpt_path $model_name $action_ensemble_temp $logging_dir $device
  # done

  # statistics evalution results
  echo "🚀 all tasks DONE! Calculating metrics..."
  echo $logging_dir
  /mnt/petrelfs/houzhi/anaconda/envs/rt_dp/bin/python tools/calc_metrics_evaluation_videos.py \
    --log-dir-root $logging_dir \
    >>$logging_dir/total.metrics
done
