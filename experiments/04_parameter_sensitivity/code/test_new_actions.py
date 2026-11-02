#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pso_rl_new import get_parameters_by_action
from rl_agent_new import QLearningAgent
import numpy as np

def test_get_parameters_by_action():
    """测试get_parameters_by_action函数"""
    print("测试get_parameters_by_action函数:")
    print("=" * 50)
    
    for action in [0, 1, 2]:
        action_names = ["固定参数", "线性递减w", "分段策略"]
        print(f"\n动作 {action} ({action_names[action]}):")
        for progress in [0.0, 0.3, 0.5, 0.7, 1.0]:
            w, c1, c2 = get_parameters_by_action(action, progress)
            print(f"  进度 {progress:.1f}: w={w:.2f}, c1={c1:.2f}, c2={c2:.2f}")

def test_ql_agent():
    """测试QLearningAgent"""
    print("\n\n测试QLearningAgent:")
    print("=" * 50)
    
    agent = QLearningAgent(actions=[0, 1, 2])
    print(f"QLearningAgent初始化成功，动作空间: {agent.actions}")
    
    print("\n测试动作选择:")
    for i in range(10):
        action = agent.choose_action((0, 0, 0, 0, 0))
        print(f"  第{i+1}次选择动作: {action}")

def test_integration():
    """测试集成"""
    print("\n\n测试集成:")
    print("=" * 50)
    
    agent = QLearningAgent(actions=[0, 1, 2])
    
    # 模拟一个完整的迭代
    progress = 0.3
    state = (1, 2, 0, 1, -1)  # 示例状态
    action = agent.choose_action(state)
    w, c1, c2 = get_parameters_by_action(action, progress)
    
    print(f"状态: {state}")
    print(f"选择的动作: {action}")
    print(f"根据动作和进度得到的参数: w={w:.2f}, c1={c1:.2f}, c2={c2:.2f}")
    
    # 模拟奖励和更新
    reward = 1.5
    next_state = (1, 1, 1, 0, 0)
    agent.update(state, action, reward, next_state)
    print(f"更新Q表成功，奖励: {reward}")

if __name__ == "__main__":
    test_get_parameters_by_action()
    test_ql_agent()
    test_integration()
    print("\n所有测试完成！")

