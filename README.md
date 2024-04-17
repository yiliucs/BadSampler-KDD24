# BadSampler-KDD24
## Additional experimental results

Our attack code: https://github.com/yiliucs/BadSampler-KDD24/blob/main/BadSampler.zip

# New Results for Reviewer msjs

Q1:

| Attack (K=1000, M=1%)    | $\Delta$ |
| ----------- | ----------- |
|Ours (Meta)|5.44%|

![fig 1](https://github.com/yiliucs/BadSampler-KDD24/blob/main/Acc1000_10.png)


Q2:

| Defences     | $\Delta$ |
| ----------- | ----------- |
|[R3]|7.68%|
|[R4]|8.97%|
|[31]|6.42%|

Q3:

 - The mentioned accuracy (loss) trajectories pertain to the test accuracy (loss) trajectories of the global model, both with and without the BadSampler attack.

![fig 1](https://github.com/yiliucs/BadSampler-KDD24/blob/main/Acc.png)

![fig 1](https://github.com/yiliucs/BadSampler-KDD24/blob/main/loss.png)

 - The validation accuracy (loss) trajectories of the local client, both with and without the BadSampler attack.

![fig 1](https://github.com/yiliucs/BadSampler-KDD24/blob/main/valacc_cifar.png)

![fig 1](https://github.com/yiliucs/BadSampler-KDD24/blob/main/valloss_cifar.png)


# Reviwer HQZq
Q1:

![fig 1](https://github.com/yiliucs/BadSampler-KDD24/blob/main/TESTattack.png)


# Reviewer msjs

Q1:

![fig 1](https://github.com/yiliucs/BadSampler-KDD24/blob/main/Acc1000.png)

Q2:

| Attack   |      Dataset      |  $\Delta$ |
|----------|:-------------:|------:|
| [33] |  FMNIST | 10.67% |
| [33] |  CIFAR-10 | 5.4% |
| Meta (Ours) |    FMNIST   |   12.96% |
| Meta (Ours) |    CIFAR-10   |   19.30% |


Q3:

| Attack   |      Defense      |  $\Delta$ |
|----------|:-------------:|------:|
| Meta (Ours) |    CONTRA   |   9.43% |
| Meta (Ours) |    DnC   |   7.65% |

Q4:

| Attack     | $\Delta$ |
| ----------- | ----------- |
|LMPA|8.21%|
|DOA|5.46%|
|OBLIVION|6.78%|
|Top-𝜅 (Ours)|10.7%|
|Meta (Ours)|13.15%|
Q5:

![fig 1](https://github.com/yiliucs/BadSampler-KDD24/blob/main/Acc.png)

![fig 1](https://github.com/yiliucs/BadSampler-KDD24/blob/main/loss.png)

Q6: 

| Method     | $\Delta$ |
| ----------- | ----------- |
|FedAvg|72.8%|
|Meta (Ours)|62.8%|

Q7:

| α     | $\Delta$ |
| ----------- | ----------- |
|0.1|21.2%|
|0.5|25.6%|
|1|27.4%|


# Reviewer uF7w

Q1:

| k     | $\Delta$ |
| ----------- | ----------- |
|16|8.71%|
|32|7.44%|
|64|7.87%|

| B     | $\Delta$ |
| ----------- | ----------- |
|128|8.64%|
|512|9.12%|
|1024|7.97%|

Q2:

| Attack     | $\mathcal{D}_{cos}$ | $H_N$|$H_D$|
| ----------- | ----------- |----------- |----------- |
|LMPA|Group1: 1.17, Group2: 0.18|Group1: 9243, Group2: 18964|Group1: 0.95, Group2: 0.26|
|DOA|Group1: 1.11, Group2: 0.78|Group1: 8697, Group2: 11352|Group1: 0.94, Group2: 0.82|
|OBLIVION|Group1: 1.07, Group2: 0.34|Group1: 8798, Group2: 12456|Group1: 0.94, Group2: 0.73|
|Top-𝜅 (Ours)|Group1: 1.09, Group2: 0.73 |Group1: 8456, Group2: 13256|Group1: 0.98, Group2: 0.67|
|Meta (Ours)|Group1: 1.12, Group2: 0.89 |Group1: 9687, Group2: 15428|Group1: 0.97, Group2: 0.96|

# Reviewer P31t

Q4:

![fig 1](https://github.com/yiliucs/BadSampler-KDD24/blob/main/TESTattack.png)


Q6:

| Symbols     | Description |
| ----------- | ----------- |
|$b$|The number of bins in the histogram|
|$B$|The local batch size|
|$\mathcal{B}$|The after-attack adversarial training batch|
|$c$|The number of malicious clients|
|$D_k$|The local dataset of the $k$-th client|
|$E_{local}$| The local training epoch|
|$H_N$|Hessian norm|
|$H_D$|Hessian direction|
|$K$| The the total number of clients|
|$M$|The proportion of the compromised clients|
|$N$|The number of local training batches|
|$q$|The proportion of client participation|
|$s$|The meat-state|
|$T$|The total number of training rounds|
|$\kappa$|The constant|
|$\eta$|The learning rate|
|$\mathcal{D}_{cos}$|The gradient cosine distance|


# Reviewer y1Gw

Q2:

| Method     | $\Delta$ |
| ----------- | ----------- |
|FedAvg|72.8%|
|Meat (Ours)|62.8%|


# Full version of our rebuttal

## Reviewer HQZq:

Q1: This paper does not discuss the persistence of the attack. For example, during FL, since only a small portion of clients is active for each round, the malicious clients might not participate in training for several rounds. I wonder whether the impact of the attack will fade away when the attack is stopped. If it is, how many rounds the impact of the attack can be sustained?

Answer 1:

- In our original manuscript, we adhered to FL's standard parameter settings and procedures. This entails that the malicious client is not necessarily to be involved in every training round but is rather selected randomly by the server.

- Furthermore, to delve deeper into the attack persistence of BadSampler, we conducted additional experiments. We instructed the malicious client to terminate training after 100 rounds, followed by normal training for another 100 rounds. Interestingly, we observed that the FL model failed to recover its normal state, experiencing a notable accuracy drop of 10%. This highlights the consistent and substantial impact of BadSampler, perpetuating damage to the FL model.

Result: https://anonymous.4open.science/r/BadSampler-KDD24-0419/README.md

Q2: Badsampler is designed based on the idea of increasing the validation accuracy. I wonder why this kind of attack cannot be detected by the defender using a validation dataset? A detail explanation is suggested here.


Answer 2:

- We acknowledge the proposed defense method utilizing the validation set, as suggested by the reviewer, represents a straightforward approach to combat BadSampler. However, it is worth noting that we have extensively evaluated FLTrust, a more sophisticated solution akin to the aforementioned method, within the original manuscript. FLTrust operates by leveraging a small dataset on the server-side to compute a trust score aimed at filtering out malicious updates. Despite its implementation, empirical results indicate that BadSampler adeptly circumvents such defenses. 
- Furthermore, in real-world application scenarios, the server lacks access to the local client's dataset information, rendering it challenging to construct a comprehensive validation set for robust detection of poisoning attacks. Typically, the creation of an effective validation set necessitates access to sensitive data such as task specifics, dataset distribution, and label information. 
- Additionally, the reinforcement learning (RL) component embedded within BadSampler continuously undermines the model's generalization ability by dynamically calculating local training accuracy and validation accuracy. This adaptive behavior renders the validation set defense method incapable of accurately discerning between benign and malicious updates.



## Reviewer msjs:

Q1: The experimental setting does not align with the proposed threat model. In the threat model, the authors argue that they follow the "production FL setting [34]" compared to the unrealistic assumptions in prior works [3,4,9,42,48]. In the experiments, they considered the large-scale FL scenario called the "cross-device" setting. The authors even criticize that the total client number (from 50 to 100) used in prior works is not practical (see Table 8)! However, they only consider a small number of total clients (K=100) which is inconsistent with [34]. [34] argues that the total number of clients should be over 1000 in cross-device setting. In general, the motivation of this paper is conflict with their experimental setting.

Answer 1:

- In fact, the most pivotal parameter influencing the practicality of the attack is the proportion of malicious clients (where we set  $M \leq 10\%$ ), aligning with the findings of [34].
- For additional clarification, we introduced a new series of experiments involving 1000 clients with 10% of them being malicious. Remarkably, the experimental results revealed that BadSampler could still induce a significant 30% drop in the accuracy of the FL model. This outcome is attributed to the substantial increase in the number of malicious clients.

Q2: Limited comparison with State-of-the-Art Baseline Attacks: One notable omission in the paper is the lack of comparison with some state-of-the-art and significant baseline attacks, FedInv[R1] and Cerberus[R2]. Including these baselines in the evaluation would have offered a more comprehensive understanding of BadSampler's relative effectiveness and positioned it more accurately within the landscape of existing poisoning attacks.

Answer 2:

- [R2] focuses on backdoor attacks, which is beyond the scope of this paper.
- [R1] only applies to a highly non-IID data setting and a malicious client ratio of 30%, which is different from the attack setting of this paper. Furthermore, the author of [R1] claims that the code is missing.
- Therefore, we chose the attack in [33] (SOTA poisoning attack) for comparison. The attack impact of this attack on the FMNIST and CIFAR-10 data sets is 10.67% and 5.4% respectively, both lower than BadSampler. Notably, our results underscore that BadSampler maintains its superiority over these advanced extreme attacks.

| Attack   |      Dataset      |  $\Delta$ |
|----------|:-------------:|------:|
| [33] |  FMNIST | 10.67% |
| [33] |  CIFAR-10 | 5.4% |
| Meta (Ours) |    FMNIST   |   12.96% |
| Meta (Ours) |    CIFAR-10   |   19.30% |


Q3: Lack of Recent State-of-the-Art Defenses for Robustness Evaluation: The paper positions FLTrust [8] as the state-of-the-art defense against poisoning attacks in federated learning systems. However, recent research has highlighted certain limitations of FLTrust, suggesting that it may not be the most effective defense mechanism under some circumstances [2]. Given these developments, it is crucial for a comprehensive evaluation of the proposed attacks' robustness to evaluate against other state-of-the-art defenses, including FLAME [R3], CONTRA [2], DeepSight [31], DnC [33], and RoseAgg [R4]. These defenses represent advancements in securing federated learning systems and might offer different levels of resistance against the BadSampler attack.

Answer 3:

- It is noted that [R3], [31], and [R4] all concentrate on backdoor or targeted attacks, which diverge from the focus of this paper.
- To further validate BadSampler's efficacy against advanced defenses, we chose CONTRA and DnC for evaluation. The experimental outcomes demonstrate that under these defenses, BadSampler induced a notable accuracy drop of 9.43% and 7.65%, respectively, on the FL model using the CIFAR-10 dataset. These results underscore the continued potent attack performance of BadSampler.

| Attack   |      Defense      |  $\Delta$ |
|----------|:-------------:|------:|
| Meta (Ours) |    CONTRA   |   9.43% |
| Meta (Ours) |    DnC   |   7.65% |

Q4: Lack of Baseline Attacks Evaluation in Non-IID Setting: The paper's evaluation of the proposed BadSampler attack primarily focuses on its effectiveness in a non-IID (non-Independently and Identically Distributed) setting, which is crucial for reflecting realistic federated learning environments. However, a notable limitation is the absence of comparative evaluations involving baseline attacks within the same non-IID setting. Such comparative analyses are crucial for understanding whether the advantages of BadSampler are consistent across various federated learning scenarios and how it stands in the landscape of poisoning attacks under the complex dynamics introduced by non-IID data.

Answer 4:

- To clarify this, we added relevant experiments. Experimental results show that BadSampler still outperforms other baselines in the non-IID setting.


| Attack     | $\Delta$ |
| ----------- | ----------- |
|LMPA|8.21%|
|DOA|5.46%|
|OBLIVION|6.78%|
|Top-𝜅 (Ours)|10.7%|
|Meta (Ours)|13.15%|

Q5: Absence of Training and Validation Loss Trajectory Visualization: It lacks a figure illustrating the trajectory of training and validation loss during the federated learning process under attack conditions. Such visualizations can offer intuitive insights into how an attack influences the learning dynamics, including evidence of catastrophic forgetting or subtle degradation in model performance over time. The paper introduces BadSampler, emphasizing its ability to exploit catastrophic forgetting by manipulating the model's generalization error through adversarial sampling. The visualization would provide a clear, empirical basis to support the claim that BadSampler effectively leads to increased generalization error, thereby validating the theoretical underpinnings of the attack strategy.

Answer 5:

- To address the reviewer's concerns, we provide an overview of BadSampler's training and validation loss trajectories on FMNIST dataset.

Q6: Limited Dataset Diversity and Scale: The evaluation of the proposed BadSampler attack is conducted using two datasets, Fashion-MNIST (F-MNIST) and CIFAR-10, which, while commonly used in machine learning research, are relatively small and simple. To better demonstrate the effectiveness and robustness of the proposed attack, especially in more challenging scenarios, it would be advantageous for the authors to include evaluations on larger and more complex datasets, such as the PURCHASE dataset (used in references [33] and [51]) and the LOAN dataset (used in [2] and [R2]). Including such datasets would test the scalability of BadSampler, assessing whether its effectiveness holds in scenarios with higher dimensional data and more complex data distributions.

Answer 6:
- To clarify this, we added relevant experiments by using our settings. Experimental results show that our attack can cause the FL model accuracy to decrease by 10% on the LOAN dataset (FedAvg: 72.8%, Ours: 62.8%).

| Method     | $\Delta$ |
| ----------- | ----------- |
|FedAvg|72.8%|
|Meta (Ours)|62.8%|



Q7: Simulation of Non-IID Setting not aligning with recent works: The paper's approach to simulating non-IID data settings for evaluating the proposed attack may not align with recent state-of-the-art works ([2], [46], [33], [34], [36]) that employ the Dirichlet distribution for a more realistic representation of data heterogeneity across clients in federated learning environments. The use of the Dirichlet distribution allows for the controlled simulation of varying degrees of data non-uniformity among clients. This would provide a more robust assessment of BadSampler's effectiveness in realistically heterogeneous federated learning.

Answer 7:

- To clarify this point, we verified the performance of BadSampler (Meta) on the CIFAR-10 dataset and set the Dirichlet parameter α = 0.1, 0.5, 1. The experimental results are shown in the following table:

| α     | $\Delta$ |
| ----------- | ----------- |
|0.1|21.2%|
|0.5|25.6%|
|1|27.4%|


Q8: Please add citations for "DPA" and "MPA" in Line 43.

Answer 8: We will add them in the revision.

Q9: Based on my knowledge of catastrophic forgetting, should it be "old tasks" in "Inspired by the phenomenon of catastrophic forgetting [15], well-designed benign updates (i.e., achieving higher training accuracy) may cause the global model to forget knowledge about new tasks, thus damaging the model’s generalization ability (Line 94 to 98)"?

Answer 9: Yes, you are right. We will revise it in the revision.


All additional experimental results and source code can be found at the anonymous link below: https://anonymous.4open.science/r/BadSampler-KDD24-0419/README.md 


## Reviewer uF7w:

W1. The contents of Table 4 cannot support the conclusions of section 6.2.4, especially for k and B. It is hoped that more experiments on different k and B can be added to reflect the influence of different hyperparameters on the attack effect.
Answer 1:

- To provide further insight into the impact of hyperparameters k and B on BadSampler's attack performance, we conducted verification experiments utilizing the CIFAR-10 dataset with varying values of k = {16, 32, 64} and B = {128, 512, 1024}. The experimental results are shown in the table below. Furthermore, the results corroborate the conclusions drawn in section 6.2.4, demonstrating consistency across different parameter settings. 

| k     | $\Delta$ |
| ----------- | ----------- |
|16|8.71%|
|32|7.44%|
|64|7.87%|

| B     | $\Delta$ |
| ----------- | ----------- |
|128|8.64%|
|512|9.12%|
|1024|7.97%|

W2. The relevant experiments in Table 6 are insufficient. The effectiveness of the proposed attack can be better demonstrated by adding the numerical results of some existing attacks.

Answer 2:

- We include the numerical results of the baseline attacks below for reviewers’ reference.

| Attack     | $\mathcal{D}_{cos}$ | $H_N$|$H_D$|
| ----------- | ----------- |----------- |----------- |
|LMPA|Group1: 1.17, Group2: 0.18|Group1: 9243, Group2: 18964|Group1: 0.95, Group2: 0.26|
|DOA|Group1: 1.11, Group2: 0.78|Group1: 8697, Group2: 11352|Group1: 0.94, Group2: 0.82|
|OBLIVION|Group1: 1.07, Group2: 0.34|Group1: 8798, Group2: 12456|Group1: 0.94, Group2: 0.73|
|Top-𝜅 (Ours)|Group1: 1.09, Group2: 0.73 |Group1: 8456, Group2: 13256|Group1: 0.98, Group2: 0.67|
|Meta (Ours)|Group1: 1.12, Group2: 0.89 |Group1: 9687, Group2: 15428|Group1: 0.97, Group2: 0.96|

- Findings: (1) In the case of the LMPA attack, the necessity to upload optimized toxicity gradients and the attack's optimization goal diverging from the global optimization direction result in significantly different values for Group 1 and Group 2 of the three indicators for this attack.  (2) Conversely, for DOA and OBLIVION attacks, which leverage catastrophic forgetting to target the FL model, their metric values closely resemble those of BadSampler.


W3. The paper emphasizes that not changing local data is one of the advantages of the attack.. However, under federated learning, the client's local data is not allowed to be accessed by other clients or servers. Therefore, this advantage emphasized in the paper is not attractive enough.
 
Answer 3:

- Indeed, within FL, the server lacks access to the clients' local dataset information, granting malicious clients the freedom to manipulate their local datasets at will [R1-R3]. However, in the context of poisoning attacks, the stealth of the attack becomes a critical factor. As elucidated in the dilemma analysis of existing attacks within the original manuscript: If a malicious client were to indiscriminately modify its local dataset to poison the FL model, such behavior could be easily detected and mitigated by vigilant defenders. 
- Therefore, this paper introduces BadSampler, which leverages clean datasets to bridge this gap. Through extensive case studies, we demonstrate that dataset modification is not a prerequisite for executing poisoning attacks in FL. This approach ensures a higher level of stealth and effectiveness in undermining FL model integrity without triggering immediate suspicion.

[R1] Arjun Nitin Bhagoji, Supriyo Chakraborty, Prateek Mittal, and Seraphin Calo.
2019. Analyzing federated learning through an adversarial lens. In Proc. of ICML.

[R2] Minghong Fang, Xiaoyu Cao, Jinyuan Jia, and Neil Gong. 2020. Local model poisoning attacks to Byzantine-robust federated learning. In Proc. of USENIX Security.

[R3] Clement Fung, Chris JM Yoon, and Ivan Beschastnikh. 2020. The limitations of federated learning in sybil settings. In Proc. of RAID.


## Reviewer rUYe

Q1: In line 121: |𝐷𝑝𝑜𝑖𝑠𝑜𝑛 | ≫ |𝐷|, is there a typo? In my understanding, it should be |𝐷𝑝𝑜𝑖𝑠𝑜𝑛 | << |𝐷|.

Answer 1:

- First, it is not a typo. In fact, I concur with the reviewer that |𝐷𝑝𝑜𝑠𝑜𝑛 | ≫ |𝐷| should ideally be |𝐷𝑝𝑜𝑠𝑜𝑛 | << |𝐷|. However, it is worth noting that many poisoning attack literatures deviate from this practical attack assumption and inject poison datasets exceeding the size of |𝐷| to enhance poisoning performance. This is indeed an important issue addressed in this article: how to successfully poison FL within the constraints of realistic usage parameters.

- Furthermore, the inclusion of this formula serves to highlight a dilemma: the adversary may attempt to enhance poisoning performance by configuring unrealistic attack parameters, which is impractical; or by modifying the dataset or gradient, which is easily detectable. To bridge this gap, this paper introduces BadSampler, a poisoning attack capable of effectively poisoning FL using only clean datasets within practical parameters.

Thanks again to the reviewer for your constructive comments.

## Reviewer P31t:

Q1: Since the authors focus on realistic adversary models, is adversaries holding a surrogate model as well as training an adversarial meta-sampler are realistic assumptions?

Answer 1:

- Firstly, it is entirely feasible for adversaries to possess a surrogate model, as substantiated by existing literature on machine learning attacks and federated learning attacks [R1-R5]. The availability of open-source deep learning model libraries makes it effortless and cost-effective for adversaries to acquire such models. Furthermore, our attack doesn't necessitate the adversary's surrogate model to have the same architecture as the model provided by the server, a fact corroborated by the findings presented in Table 9 of the original manuscript.

- Additionally, it's realistic for adversaries to train adversarial metasamplers. In the realm of federated learning, clients wield autonomous control over the local training process, allowing a compromised client to surreptitiously develop an adversarial sampler, thus exerting influence over the local training dynamics [R6-R7].

[R1] Qin Y, Xiong Y, Yi J, et al. Training meta-surrogate model for transferable adversarial attack[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2023, 37(8): 9516-9524.

[R2] Sun X, Cheng G, Li H, et al. Exploring effective data for surrogate training towards black-box attack[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022: 15355-15364.

[R3] Li J, Rakin A S, Chen X, et al. Ressfl: A resistance transfer framework for defending model inversion attack in split federated learning[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022: 10194-10202.

[R4] Zheng T, Li B. Poisoning attacks on deep learning based wireless traffic prediction[C]//IEEE INFOCOM 2022-IEEE Conference on Computer Communications. IEEE, 2022: 660-669.

[R5] Nguyen T D, Nguyen T A, Tran A, et al. Iba: Towards irreversible backdoor attacks in federated learning[J]. Advances in Neural Information Processing Systems, 2024, 36.

[R6] Fang M, Cao X, Jia J, et al. Local model poisoning attacks to {Byzantine-Robust} federated learning[C]//29th USENIX security symposium (USENIX Security 20). 2020: 1605-1622.

[R7] Shejwalkar V, Houmansadr A, Kairouz P, et al. Back to the drawing board: A critical evaluation of poisoning attacks on production federated learning[C]//2022 IEEE Symposium on Security and Privacy (SP). IEEE, 2022: 1354-1371.

Q2: BadSampler is based on adversarial data sampling strategies. Could this be easily mitigated by implementing TEE on client side, since it provides isolated execution of code and handling the data?

Answer 2:

We concur with the reviewer's acknowledgment of the potential security benefits offered by TEEs, which can safeguard code and data integrity. However, we provide the following justifications to clarify why deploying TEEs on the client side to defend against poisoning attacks in FL may not be practical:
- Cost and scalability: TEEs represent mature yet expensive hardware solutions, which may not be feasible for widespread deployment across thousands of FL clients, particularly in cross-device scenarios. The high cost associated with TEE implementation renders it impractical for large-scale FL systems.


Q3: In Section 3.1, while explaining generalization error, is high bias attributed to underfitting, i.e., the model being used is not robust enough to produce an accurate prediction? The generalization error is attributed to the opposite, overfitting, where ML models become too complex and start to fit the training data. This contradicts lines 231-235.

Answer 3:

- After conducting thorough reviews, it has come to our attention that an inaccurate explanation was inadvertently included in the original manuscript. To rectify this issue and provide clarity, we have made the following modifications: “According to the above definition, the bias may be smaller when the variance is larger, which means that the training error of the model is small and the verification error may be large. When the above situation occurs, the model is likely to fall into the catastrophic forgetting problem.” Thanks again to the reviewer for pointing it out.


Q4: In Table 1, the test accuracy of the CIFAR10 models is suspiciously low, especially for the ResNet-18 model trained for CIFAR-10. Even baseline FL models [26] reach test accuracy 80% in CIFAR10, why have the authors reported only 67% while achieving training accuracy at 92%? Similar inconsistencies with previous work are observed in Table 5, ResNet-18 model for CIFAR10.

Answer 4:

- We firmly believe that this phenomenon is reasonable, as evidenced by the varying accuracy results reported across different literature, particularly in the context of FL. These discrepancies in accuracy can be attributed to a multitude of factors, including the number of training epochs, the experimental setup, the number of clients involved, among others. You can observe the test accuracy curve of CIFAR-10 in the anonymous link we provide. Additionally, we will repeat the experiments in the future to correct the above table.

Link: https://anonymous.4open.science/r/BadSampler-KDD24-0419/README.md

Q5: In experimental results, did you find a particular reason why top-k and meta have a high difference in their impact against simpler models, compared to complex models?

Answer 5:

- We appreciate the insightful questions raised by the reviewers. It is our conjecture that the parameter distributions of simpler models are particularly vulnerable to the ordering of adversarial data samples, a conclusion that resonates with findings in [R8]. In essence, significant shifts in parameter distributions can result in catastrophic forgetting. For instance, even very large language models undergoing minor fine-tuning may exhibit mild instances of catastrophic forgetting and hallucination issues. This underscores the delicate balance between model robustness and susceptibility to adversarial manipulation.

[R8] Shumailov I, Shumaylov Z, Kazhdan D, et al. Manipulating sgd with data ordering attacks[J]. Advances in Neural Information Processing Systems, 2021, 34: 18021-18032.


Q6: Due to the heavy notation, I suggest the authors put a notation table to increase readability.

Answer 6:

- To make it easier for readers to follow, we have added the following table to summarize the main mathematical symbols used in this paper as follows:

| Symbols     | Description |
| ----------- | ----------- |
|$b$|The number of bins in the histogram|
|$B$|The local batch size|
|$\mathcal{B}$|The after-attack adversarial training batch|
|$c$|The number of malicious clients|
|$D_k$|The local dataset of the $k$-th client|
|$E_{local}$| The local training epoch|
|$H_N$|Hessian norm|
|$H_D$|Hessian direction|
|$K$| The the total number of clients|
|$M$|The proportion of the compromised clients|
|$N$|The number of local training batches|
|$q$|The proportion of client participation|
|$s$|The meat-state|
|$T$|The total number of training rounds|
|$\kappa$|The constant|
|$\eta$|The learning rate|
|$\mathcal{D}_{cos}$|The gradient cosine distance|



## Reviewer y1Gw:

Q1: In the sampling process, accessing data from other clients in FL is not feasible. However, I am uncertain whether a sampling method akin to Equation 7 truly meets the author's criteria.

Answer 1:

- Equation 7 suggests that the adversary can conduct adversarial sampling on compromised clients rather than the entire dataset. We acknowledge that the misuse of symbol D may have contributed to the concerns raised by the reviewer. To address this issue and provide clarity, we propose the following modification to Equation 7:

$${\mathcal{H}_{bad}} = Top - \kappa (er{r_i},{D_k},B).$$


Q2: I'm slightly concerned that the validation using only two datasets, F-MNIST and CIFAR10, may not adequately reflect the generalizability of this approach.

Answer 2:

- To clarify this, we added relevant experiments by using our settings. Experimental results show that our attack can cause the FL model accuracy to decrease by 10% on the LOAN dataset (FedAvg: 72.8%, Ours: 62.8%). Due to time constraints, we will include more baseline attacks and advanced defenses on the LOAN dataset in the revised version.


| Method     | $\Delta$ |
| ----------- | ----------- |
|FedAvg|72.8%|
|Meat (Ours)|62.8%|



