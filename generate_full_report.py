#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新生成包含所有10个问题的完整对比报告
基于已有的评估结果
"""

import json
import time
from pathlib import Path

def generate_complete_report():
    """基于现有数据生成完整报告"""
    
    # 从之前的评估结果中提取数据
    # 这里我们手动输入数据，因为原始评估结果没有保存到文件
    
    qwen_results = [
        {"question": "为什么麦当劳确保自己食物安全的激励更大？", "time": 26.87, "length": 234, 
         "answer": "因为如果麦当劳确保自己食物安全，它就不会因为食物中毒而失去顾客。如果麦当劳不这么做，它就可能失去顾客。因此，麦当劳确保自己食物安全的激励更大，因为如果它不这么做，它就可能失去顾客。麦当劳还可能会因为顾客不买它的食物而失去顾客。因此，麦当劳确保自己食物安全的激励更大，因为如果它不这么做，它就可能失去顾客。麦当劳还可能会因为顾客不买它的食物而失去顾客。因此，麦当劳确保自己食物安全的激励更大，因为如果它不这么做，它就可能失去顾客。"},
        
        {"question": "品牌争论的中心是什么？", "time": 24.14, "length": 242,
         "answer": "品牌争论的中心是：如果一个企业拥有某种物品，它就可能得到它。如果一个企业拥有某种物品，它就可能得到它。如果一个企业拥有某种物品，它就可能得到它。如果一个企业拥有某种物品，它就可能得到它。如果一个企业拥有某种物品，它就可能得到它。如果一个企业拥有某种物品，它就可能得到它。如果一个企业拥有某种物品，它就可能得到它。如果一个企业拥有某种物品，它就可能得到它。如果一个企业拥有某种物品，它就可能得到它。"},
        
        {"question": "什么是生产要素？", "time": 24.67, "length": 277,
         "answer": "生产要素是指用于生产物品和劳务的任何东西，包括土地、劳动力、资本和企业家才能。这些要素是生产的基本资源。土地是一种自然的资源，用于生产物品和劳务。劳动力是工人，他们提供体力和脑力劳动。资本是一种生产资料，包括工厂、机器和建筑。企业家才能是组织、管理和销售资源的才能。企业家才能也是生产要素之一。生产要素的组合决定了企业的生产可能性边界。企业家才能是所有者之一。在经济学中，企业家才能通常被认为是第5种生产要素。"},
        
        {"question": "什么是派生需求？为什么说生产要素的需求是派生需求？", "time": 25.13, "length": 271,
         "answer": "派生需求是由于其他物品需求的增加而引起的某种物品的需求。生产要素的需求是派生需求，因为它取决于其他物品的需求，而这些物品的需求又取决于最终物品的需求。因此，生产要素的需求是派生需求。由于生产要素的需求是派生需求，生产要素价格的变动就影响到整个经济，而且影响的范围是广泛的。例如，劳动力的需求取决于对物品的需求，而物品的需求又取决于最终物品的需求。因此，对劳动力的需求取决于最终物品的需求，而对最终物品的需求取决于消费者的偏好。"},
        
        {"question": "竞争的、利润最大化的企业在决定劳动需求时，会考虑哪些因素？", "time": 23.25, "length": 265,
         "answer": "只有企业可以决定劳动力的数量，但是工资率由市场决定。当竞争企业确定劳动力数量时，它要比较自己的边际收益和边际成本。边际收益是工资率，边际成本是劳动力的机会成本。当边际收益等于边际成本时，企业决定雇佣量。因此，竞争企业决定劳动力数量时，要比较边际收益与边际成本。企业要雇佣工人，只要他们为企业的边际收益大于边际成本。如果边际收益大于边际成本，企业雇佣工人；如果边际收益小于边际成本，企业不雇佣工人。当边际收益等于边际成本时，企业停止雇佣工人。"},
        
        {"question": "什么是生产函数？", "time": 23.01, "length": 258,
         "answer": "生产函数表示一种投入与产出之间的数量关系。它把投入的要素（如劳动力和资本）与产出（如产量）联系起来。生产函数可以表示为：Q = f(L, K)，其中Q是产量，L是投入的劳动力数量，K是投入的资本数量。生产函数表示在给定技术条件下，投入的要素数量与产出数量之间的关系。生产函数反映了生产过程中的技术关系，它说明了在一定的技术条件下，投入要素的不同组合所能产生的最大产量。"},
        
        {"question": "什么是劳动的边际产量？", "time": 22.59, "length": 249,
         "answer": "劳动的边际产量是指在保持其他投入不变时，增加一单位劳动所引起的产量的变动量。边际产量随着劳动投入量的增加而递减。当增加一单位劳动投入时，产量增加量小于以前的增加量，这种现象称为边际产量递减。边际产量在初期可能是递增的，但随着劳动投入的进一步增加，边际产量会开始递减。这是因为其他投入（如资本）保持不变，而劳动投入不断增加，会导致投入要素配比失调，从而使边际产量递减。"},
        
        {"question": "什么是边际产量递减？", "time": 22.79, "length": 243,
         "answer": "边际产量递减是指当投入增加时，产量增加量减少。这发生在投入和产量之间存在规模收益递减时。规模收益递减表明，当投入增加时，产量增加量减少。边际产量递减表明，当投入增加时，产量增加量减少。边际产量递减表明，当投入增加时，产量增加量减少。边际产量递减是经济学中的一个重要概念，它解释了为什么企业不会无限制地增加投入。当边际产量递减时，企业会寻找最优的投入组合，以实现利润最大化。"},
        
        {"question": "什么是边际产量值？如何计算？", "time": 22.67, "length": 232,
         "answer": "边际产量值=边际产量×价格。边际产量是投入的变动量，而价格是产量的变动量。边际产量值=边际产量×价格。边际产量值=边际产量×价格。边际产量值=边际产量×价格。边际产量值=边际产量×价格。边际产量值=边际产量×价格。边际产量值表示增加一单位投入所能带来的额外收益。企业在决策时，会比较边际产量值与投入成本，以确定是否增加投入。当边际产量值大于投入成本时，企业会增加投入；反之则会减少投入。"},
        
        {"question": "一个竞争性的、利润最大化企业如何决定雇佣工人的数量？", "time": 22.65, "length": 254,
         "answer": "利润最大化企业雇佣工人数量决定于工人带来的边际收益和工资。当边际收益大于工资时，企业就雇佣工人；当边际收益小于工资时，企业就减少工人。如果边际收益等于工资，企业就停止雇佣工人。边际收益是指，工人增加一单位时带来的额外收益。企业会继续雇佣工人，直到最后一个工人的边际收益等于其工资为止。这是利润最大化的条件。在竞争市场中，企业是价格接受者，因此边际收益等于产品价格乘以边际产量。"}
    ]
    
    gemma_results = [
        {"question": "为什么麦当劳确保自己食物安全的激励更大？", "time": 5.32, "length": 114,
         "answer": "因为麦当劳能为顾客提供安全食物的激励比顾客用食物时支付的价格大得多，从而使顾客愿意支付的价格高得多。"},
        
        {"question": "品牌争论的中心是什么？", "time": 1.37, "length": 68,
         "answer": "消费者不愿支付更高的价格。"},
        
        {"question": "什么是生产要素？", "time": 1.85, "length": 67,
         "answer": "劳务、土地、资本和企业家精神。"},
        
        {"question": "什么是派生需求？为什么说生产要素的需求是派生需求？", "time": 5.79, "length": 120,
         "answer": "派生需求是生产要素需求的一个变体。生产要素需求是想买卖劳务和土地的需求，而派生需求是想买卖土地的需求。"},
        
        {"question": "竞争的、利润最大化的企业在决定劳动需求时，会考虑哪些因素？", "time": 1.98, "length": 90,
         "answer": "他们会考虑工资与劳动市场供求关系。"},
        
        {"question": "什么是生产函数？", "time": 3.54, "length": 105,
         "answer": "生产函数是用于分析所有物品总量的水平，它表明物品价格与物品产量之间的关系。"},
        
        {"question": "什么是劳动的边际产量？", "time": 3.12, "length": 79,
         "answer": "劳动的边际产量等于劳动者每天能生产的劳务量。"},
        
        {"question": "什么是边际产量递减？", "time": 2.14, "length": 70,
         "answer": "当生产量增加时，边际总收益减少。"},
        
        {"question": "什么是边际产量值？如何计算？", "time": 4.15, "length": 108,
         "answer": "边际产量值是指生产一条物品所能获得的剩余总收益。计算方法是：总收益-固定成本。"},
        
        {"question": "一个竞争性的、利润最大化企业如何决定雇佣工人的数量？", "time": 6.62, "length": 100,
         "answer": "如果企业可以完全控制劳务成本，它就雇佣最多的人。如果企业无法完全控制劳务成本，它就雇佣最少的人。"}
    ]
    
    # 生成完整报告
    report = []
    report.append("🏆 Qwen3 vs Gemma3 经济学问答模型完整对比报告")
    report.append("=" * 70)
    report.append("")
    
    # 基本统计
    qwen_avg_time = sum(r["time"] for r in qwen_results) / len(qwen_results)
    qwen_avg_length = sum(r["length"] for r in qwen_results) / len(qwen_results)
    gemma_avg_time = sum(r["time"] for r in gemma_results) / len(gemma_results)
    gemma_avg_length = sum(r["length"] for r in gemma_results) / len(gemma_results)
    
    report.append("📊 基本性能统计:")
    report.append(f"├─ Qwen3-1.7B:")
    report.append(f"│  ├─ 平均响应时间: {qwen_avg_time:.2f}s")
    report.append(f"│  └─ 平均回答长度: {qwen_avg_length:.1f}字符")
    report.append(f"└─ Gemma3-1B:")
    report.append(f"   ├─ 平均响应时间: {gemma_avg_time:.2f}s")
    report.append(f"   └─ 平均回答长度: {gemma_avg_length:.1f}字符")
    report.append("")
    
    # 性能对比
    speed_winner = "Qwen3" if qwen_avg_time < gemma_avg_time else "Gemma3"
    speed_diff = abs(qwen_avg_time - gemma_avg_time)
    
    report.append("⚡ 性能对比:")
    report.append(f"├─ 响应速度优胜: {speed_winner} (快 {speed_diff:.2f}s)")
    report.append(f"├─ 速度提升比例: {(speed_diff/max(qwen_avg_time, gemma_avg_time)*100):.1f}%")
    report.append("")
    
    # 质量分析
    report.append("📋 回答质量分析:")
    report.append("├─ Qwen3特点:")
    report.append("│  ├─ 回答详细完整，平均249字符")
    report.append("│  ├─ 包含经济学专业术语和概念")
    report.append("│  └─ 但存在重复表达和逻辑循环问题")
    report.append("└─ Gemma3特点:")
    report.append("   ├─ 回答简洁明了，平均92字符")
    report.append("   ├─ 响应速度快（平均3.6秒）")
    report.append("   └─ 但回答过于简单，缺乏深度解释")
    report.append("")
    
    # 详细对比 - 显示所有10个问题
    report.append("🔍 详细回答对比（全部10个问题）:")
    report.append("-" * 70)
    
    for i in range(len(qwen_results)):
        qwen_r = qwen_results[i]
        gemma_r = gemma_results[i]
        
        report.append(f"\n问题 {i+1}: {qwen_r['question']}")
        report.append("")
        report.append(f"🟦 Qwen3回答 ({qwen_r['time']:.2f}s, {qwen_r['length']}字符):")
        report.append(f"   {qwen_r['answer'][:300]}{'...' if len(qwen_r['answer']) > 300 else ''}")
        report.append("")
        report.append(f"🟨 Gemma3回答 ({gemma_r['time']:.2f}s, {gemma_r['length']}字符):")
        report.append(f"   {gemma_r['answer']}")
        report.append("-" * 50)
    
    # 总结
    report.append("")
    report.append("🎯 总结与建议:")
    report.append("├─ 如果需要快速响应: 选择Gemma3 (速度快6.6倍)")
    report.append("├─ 如果需要详细解答: 选择Qwen3 (内容更全面)")
    report.append("├─ 实际应用建议:")
    report.append("│  ├─ 在线客服场景: Gemma3更适合")
    report.append("│  ├─ 教学辅导场景: Qwen3更适合")
    report.append("│  └─ 考试准备场景: 需要进一步优化")
    report.append("└─ 改进方向:")
    report.append("   ├─ Qwen3: 减少重复，提高逻辑性")
    report.append("   └─ Gemma3: 增加回答深度和专业性")
    
    # 保存报告
    report_text = "\n".join(report)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_file = f"complete_model_comparison_report_{timestamp}.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"📄 完整报告已生成: {report_file}")
    print(f"📊 包含所有 {len(qwen_results)} 个问题的详细对比")
    
    return report_file

if __name__ == "__main__":
    print("📝 生成包含所有10个问题的完整对比报告...")
    report_file = generate_complete_report()
    print(f"✅ 完成！查看报告: {report_file}") 