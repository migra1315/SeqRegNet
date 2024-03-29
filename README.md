  提出了一种U-LSTM算法用于无监督端到端肺部4DCT医学影像配准。U-LSTM以最大吸气相-最大呼气相周期的肺部4DCT图像作为输入，端到端非迭代地输出影像之间对应的形变场。算法综合UNet和LSTM，构建了一个4级、降采样+升采样的U型网络架构。网络的第1级由一组3D  CNN卷积核构成，在降采样路径用于提取影像的表层结构特征，在升采样路径实现根据前置信息推理最终形变场。网络的2-4级分别由一组LSTM网络构成，相较于CNN，LSTM在提取影像的结构特征之外，更能关注到影像的周期运动特性，以实现更为精确的配准。在降采样路径和升采样路径之间，相应的层级采用跳层连接，实现低分辨率高维特征和高分辨率低维特征的信息充分利用。  

针对现有的医学影像配准算法存在的配准精度低、推理速度慢的问题， U-LSTM创新性地将肺部的周期运动规律作为隐式的先验条件用于配准，构建U形架构保证了配准网络信息的全面性，以深度学学习端到端推理的方式实现了快速，精准，适用于临床穿刺手术的医学影像配准。

![pipline](.\README\pipline.png)

算法在公开数据集Dir-lab上进行测试，经过配准后的共10组图像目标配准误差（TRE）从8.46mm可以降至1.16mm，其中最低的目标配准误差仅0.96mm。相较于其他基于深度学习的配准算法（GroupRegNet,2021 1.26mm；Lung-CRNet,2021 1.56mm； LungRegNet  1.59mm），我们的算法在配准精度上也更具优势。  