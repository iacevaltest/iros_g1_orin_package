#!/usr/bin/env bash
# ROS 2 / DDS environment for the robot-side stack.
#
#   source config/ros_dds_env.sh
#
# Activating the conda environment alone is NOT enough: it leaves
# CYCLONEDDS_URI, CYCLONEDDS_HOME and PYTHONPATH unset, and the ROS distro
# it provides may differ from any system-wide install.

# Let RPATH resolve; a stale LD_LIBRARY_PATH breaks the conda stack.
unset LD_LIBRARY_PATH

# Source build of CycloneDDS, used by the direct robot link.
export CYCLONEDDS_HOME="${CYCLONEDDS_HOME:-$HOME/cyclonedds/install}"
export CYCLONEDDS_URI="${CYCLONEDDS_URI:-file://$HOME/cyclonedds_ws/cyclonedds.xml}"

# Deliberately NOT rmw_cyclonedds_cpp.
#
# Two incompatible builds of libddsc can coexist: conda-forge's (has
# shm_set_data_state, which librmw_cyclonedds_cpp.so needs) and the source
# build (lacks it, but is what the cyclonedds Python package needs). The
# Python binding loads first via ctypes, so rmw_cyclonedds_cpp then dies with
# "undefined symbol: shm_set_data_state". FastRTPS sidesteps libddsc entirely.
#
# THIS MUST MATCH IN EVERY ROS 2 PROCESS. A mismatch is silent: both sides
# start cleanly and the topics simply never connect.
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

export AMENT_PREFIX_PATH="${CONDA_PREFIX:-$AMENT_PREFIX_PATH}"
export ROS_DISTRO="${ROS_DISTRO:-humble}"
export PYTHONPATH="$HOME/GR00T-WholeBodyControl${PYTHONPATH:+:$PYTHONPATH}"

echo "[ros_dds_env] RMW=$RMW_IMPLEMENTATION  URI=$CYCLONEDDS_URI"
