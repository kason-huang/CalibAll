# CalibAll 代码走读配图生成记录

日期：2026-10-17
方式：内置 Image Gen（非 CLI）。以下图片为教学示意，不是 Panda 实拍或测量结果。

## 图 1 原始提示词

Use case: scientific-educational. Create a polished landscape Chinese infographic for a CalibAll code walkthrough. Cream background, dark teal headings, teal arrows, muted sage panels; crisp technical editorial design, readable large Chinese text, no decorative clutter, no invented performance metrics. Title exact: "CalibAll：两条业务链". Top lane title "标定链". Main left to right sequence: "视频 + 关节角 + 机器人模型" → "Web 交互 / SAM 提示" → "粗估：CoTracker + FK + PnP" → "细化：可微渲染" → "内外参 / YAML / 叠图视频". Label below coarse box "CoarseInit", below refinement "Refinement". Bottom lane title "标注链". Sequence "已保存的标定 YAML + 机器人状态" → "TCP 位姿与投影" → "Mask / BBox" → "JSON / LeRobot". A connecting arrow from top final output down to bottom YAML input explicitly labels "标定结果复用". Use small robot arm, camera and image icons illustrating context. Footer exact "先标定，再批量标注；Web 跑通不等于全部标签已导出". Logical diagram, not a screenshot. Wide 3:2 or16:9.

## 图 2 原始提示词

Use case: scientific-educational. Create a landscape Chinese educational infographic for CalibAll explaining temporal PnP initialization. Same style: warm off-white, dark teal, sage panels, orange correspondence dots. Title exact "粗标定：两条轨迹，求一个相机位姿". Upper lane: three video film frames of the same robot gripper moving, each with orange tracking dot, label "视频 → CoTracker → 二维轨迹 pᵢ". Lower lane: simple 3D robot base coordinate axes, one gripper tracked at three different positions, label "关节角 → 正向运动学 FK → 三维法兰轨迹 Pᵢ". Dashed vertical connectors pair equal-time upper and lower samples, labels "同一时刻配对". The two lanes converge at a box "PnP + 已知内参 K", then arrow to a camera position/orientation icon labelled "粗外参 R₀、t₀". Include precise equation "pᵢ ≈ π(K(RPᵢ+t))". Footer in orange band exact "点选特征未必等于法兰原点，因此这里只提供粗初值". A tiny legend says "相机相对基座固定". Avoid implying all points on a rigid body have identical trajectories. No network training diagram, no invented numbers. Highly legible Chinese and mathematical symbols.

## 图 3 原始提示词

Use case: scientific-educational. Create a polished landscape Chinese infographic explaining CalibAll differentiable mask refinement; same cream and teal academic editorial style. Title exact "细化：固定机器人，调整相机". Layout a large clockwise four-step optimization loop, arrows clear: 1 left upper "参考 Mask M" with a white robot silhouette on dark field, small label "SAM 分割 / 人工确认"; 2 right upper "渲染 Mask M̂" with outlined three-dimensional robot arm and virtual camera, small label "网格 + 关节角 + 当前外参"; both masks converge to 3 bottom right "比较区域差异" label equation "L = MSE + 0.5 × (1 − IoU)"; arrow to 4 bottom left "梯度 + Adam" small label "更新相机 R、t"; return arrow goes to rendering box 2, NEVER to reference mask box 1. Draw reference mask as static input separately outside update loop if clearer. Bottom two contrasting concise legend boxes: "固定：K、关节角、机器人网格、参考 Mask" and "优化：相机 6 自由度位姿". Small side-by-side comparison silhouettes show orange model misaligned before, teal model matching after; label "仅示意，不是实测". Footer exact "轮廓拟合改善 ≠ 独立标定精度证明". No robot joint updates, no SAM weight training. Big legible Chinese, correct mathematical labels, clean arrow topology.

## 图 1 定向修订提示词

Edit this infographic to correct its technical content, preserving overall two-lane structure, Chinese headings, typography, cream/teal/sage palette and layout. Remove ALL checkerboards from every scene: CalibAll uses robot geometry, not a chessboard calibration target. Replace them with plain tabletop/background. In bottom Mask/BBox panel show bounding boxes and masks ONLY on robot arm and gripper; remove the blue checkerboard label and box and replace label with 'gripper' on the gripper. In upper coarse panel the little internal CoTracker → FK → PnP chain is wrong: replace it with TWO parallel nodes 'CoTracker' and 'FK' both with arrows into 'PnP'. Correct coarse class name exactly 'CoarseInit'. No new claims. Other text and architecture unchanged.

## 图 3 定向修订提示词

Edit only the gradient update box number4, preserving every other part of this scientific infographic. Remove the equation '(R,t) ← (R,t) − η · Adam(∇L)' because the code optimizes an SE(3) six-vector, not raw rotation matrix entries. Replace that equation area with two exact readable lines 'SE(3) 六维参数 dof' and '反向传播 → Adam 更新 → exp 映射得到 R、t'. Change the small text '更新相机 R、t' to '更新相机位姿参数'. Keep all other boxes, arrows, loss formula, legends, colors, Chinese text and composition unchanged.
