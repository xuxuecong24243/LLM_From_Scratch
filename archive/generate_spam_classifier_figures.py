"""历史绘图脚本（原 1.py）：需要原 Linux 字体路径，输出到 outputs/。"""


def main():
    from pathlib import Path
    import zipfile
    import tempfile
    import re
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties, fontManager
    from fontTools.ttLib import TTCollection
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
    from matplotlib.backends.backend_pdf import PdfPages

    from project_paths import OUTPUT_DIR
    ROOT = OUTPUT_DIR / "spam_classifier_figures"
    ROOT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'mathtext.fontset': 'cm', 'svg.fonttype': 'path', 'pdf.fonttype':42})
    font_cache = Path(tempfile.mkdtemp(prefix='gpt2_figure_fonts_'))
    for style in ['Regular', 'Bold']:
        collection=TTCollection('/usr/share/fonts/opentype/noto/NotoSerifCJK-'+style+'.ttc')
        selected=next(f for f in collection.fonts if f['name'].getDebugName(1)=='Noto Serif CJK SC')
        path=font_cache/('NotoSerifCJKSC-'+style+'.otf')
        selected.save(path); fontManager.addfont(path)
    fontManager.addfont('/usr/share/texmf/fonts/opentype/public/lm/lmmono10-regular.otf')
    REG = FontProperties(family=['Nimbus Roman','Noto Serif CJK SC'])
    BOLD = FontProperties(family=['Noto Serif CJK SC'],weight='bold')
    MONO = FontProperties(family=['Latin Modern Mono','Noto Serif CJK SC'])
    ROMAN = FontProperties(family=['Nimbus Roman','Noto Serif CJK SC'])
    for prop in [REG,BOLD,MONO,ROMAN]: prop.set_math_fontfamily('cm')
    BLUE='#2D6892'; INK='#162530'; PALE='#EDF5FB'; BORDER='#8BB1CD'; GRAY='#50616D'
    figures=[]

    def text(ax,x,y,s,size=16,bold=False,ha='left',font=None,color=INK):
        prop=font or (BOLD if bold else REG)
        parts=[p for p in re.split(r'(\$.*?\$)',s) if p]
        if len(parts)==1:
            return ax.text(x,y,s,fontsize=size,fontproperties=prop,
                           va='center',ha=ha,color=color,linespacing=1.5)
        # Keep Chinese outside mathtext: mathtext does not use CJK font fallbacks.
        renderer=ax.figure.canvas.get_renderer()
        artists=[ax.text(0,y,p,fontsize=size,fontproperties=prop,va='center',ha='left',color=color) for p in parts]
        scale=ax.transData.transform((1,0))[0]-ax.transData.transform((0,0))[0]
        widths=[a.get_window_extent(renderer).width/scale for a in artists]
        start=x-({'left':0,'center':.5,'right':1}[ha])*sum(widths)
        for a,w in zip(artists,widths):a.set_x(start);start+=w
        return artists

    def box(ax,x,y,w,h,fill=PALE,edge=BORDER,dashed=False):
        p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.025,rounding_size=0.9',
                        fc=fill,ec=edge,lw=1.05,linestyle='--' if dashed else '-')
        ax.add_patch(p); return p

    def arrow(ax,x1,y1,x2,y2,connectionstyle='arc3,rad=0'):
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=13,
                                    lw=1,color=INK,connectionstyle=connectionstyle))

    def panel(ax,x,y,w,h,title):
        box(ax,x,y,w,h,fill='white',edge=BLUE)
        box(ax,x,y,w,6,fill=BLUE,edge=BLUE)
        text(ax,x+3,y+3,title,18,bold=True,color='white')

    def tag(ax,x,y,n,title):
        ax.add_patch(Circle((x,y),1.55,color=BLUE))
        text(ax,x,y,str(n),13,bold=True,ha='center',color='white')
        text(ax,x+3,y,title,17,bold=True)

    def page(title,subtitle,num):
        f,ax=plt.subplots(figsize=(16,12),dpi=150)
        f.subplots_adjust(0,0,1,1); ax.set(xlim=(0,160),ylim=(120,0));ax.axis('off')
        p=box(ax,5,3,150,9,fill='none',edge='#C5DCEA')
        g=np.linspace(0,1,80)[:,None,None]
        a=np.array([.94,.97,.99])[None,None,:]; b=np.array([.84,.91,.96])[None,None,:]
        arr=np.repeat(a*(1-g)+b*g,8,axis=1)
        im=ax.imshow(arr,extent=(5,155,12,3),aspect='auto',zorder=-1);im.set_clip_path(p)
        text(ax,80,7.5,title,25,bold=True,ha='center')
        text(ax,80,15,subtitle,14,ha='center',color=GRAY)
        text(ax,6,117,'依据上传脚本绘制 · 未展示数值运行结果',11,color=GRAY)
        text(ax,154,117,f'{num} / 3',12,ha='right',font=ROMAN,color=GRAY)
        figures.append(f);return f,ax

    # Figure 1: exact Dataset/DataLoader behavior, no fabricated token IDs.
    f,ax=page('GPT-2 短信分类：数据如何变成一个批次',
              'SpamDataset 负责单条样本；DataLoader 自动取样并堆叠成批次',1)
    panel(ax,6,21,72,85,'一、编码与长度统一')
    panel(ax,82,21,72,85,'二、按索引取样与自动组批')
    tag(ax,10,32,1,'读取短信与标签')
    box(ax,10,36,64,12)
    text(ax,13,39.5,'pd.read_csv(csv_file)',16,font=MONO)
    text(ax,13,44.5,'列名：text（文本）与 Label（整数 0/1）',15)
    arrow(ax,42,48.5,42,52)
    box(ax,10,52,64,11)
    text(ax,13,55.5,'tokenizer.encode(text)',16,font=MONO)
    text(ax,13,60,'每条短信 → 长度不同的 token ID 列表',15)
    arrow(ax,42,63.5,42,67)
    tag(ax,10,71,2,'确定统一序列长度 T')
    text(ax,13,77,'训练集：T = 最长编码长度',16)
    text(ax,13,82,'验证集 / 测试集：沿用训练集的 T',16)
    box(ax,10,87,64,14)
    text(ax,13,90.5,'指定 T 时先截断：encoded_text[:T]',14,font=MONO)
    text(ax,13,95.5,'右侧补齐：原列表 + [50256] × 补齐数量',15)

    tag(ax,86,32,3,'dataset[i] 调用 __getitem__(i)')
    box(ax,86,37,64,16)
    text(ax,89,41,'token_ids：torch.long，形状 [T]',15)
    text(ax,89,47,'label：torch.long，标量，形状 []',15)
    arrow(ax,118,53.5,118,57)
    tag(ax,86,61,4,'DataLoader 自动收集多条样本')
    box(ax,86,66,64,11)
    text(ax,89,69.5,'默认 collate：分别堆叠输入和标签',15)
    text(ax,89,74,'训练：batch_size=8，shuffle=True',14,font=MONO)
    arrow(ax,118,77.5,118,81)
    box(ax,86,81,64,18)
    text(ax,89,85,'input_batch',16,font=MONO)
    text(ax,146,85,r'$[B,T]$',20,ha='right')
    text(ax,89,92,'target_batch',16,font=MONO)
    text(ax,146,92,r'$[B]$',20,ha='right')
    text(ax,89,97,'B：批次样本数；T：每条样本的词元数',13,color=GRAY)
    text(ax,7,110,'训练丢弃不足 8 条的末批；验证 / 测试保留末批，故 B ≤ 8。',14)
    text(ax,7,113.8,'模型最多支持 1024 个位置；脚本未检查 T 是否超过此上限。',13,color=GRAY)

    # Figure 2: directly evidenced output head, freezing scope, exact readout.
    f,ax=page('GPT-2 短信分类：分类头、张量维度与微调范围',
              '当前配置：GPT-2 Small · 12 个 Transformer 块 · 隐藏维度 768 · 两个类别',2)
    panel(ax,6,21,69,87,'一、前向数据流与参数更新范围')
    panel(ax,80,21,74,87,'二、线性分类与最后位置读出')
    box(ax,10,30,61,8)
    text(ax,40.5,34,r'输入 token IDs  $X\in\mathbb{N}^{B\times T}$',17,ha='center')
    arrow(ax,40.5,38.5,40.5,42)
    box(ax,10,42,61,20,fill='#F7FAFC',dashed=True)
    text(ax,13,46,'GPT 主干：内部实现见 chapter04.py',15,bold=True)
    text(ax,13,52,'词元 / 位置嵌入、前 11 块：冻结',15)
    text(ax,13,58,'最后 1 个 Transformer 块：可训练',15,color=BLUE)
    arrow(ax,40.5,62.5,40.5,66)
    box(ax,10,66,61,10)
    text(ax,13,69,'final_norm：可训练',16,font=MONO,color=BLUE)
    text(ax,13,73.3,r'输出隐藏状态  $H\in\mathbb{R}^{B\times T\times768}$',16)
    arrow(ax,40.5,76.5,40.5,80)
    box(ax,10,80,61,10)
    text(ax,13,83.4,'新分类头：Linear(768, 2)',16,font=MONO,color=BLUE)
    text(ax,13,87.4,'新初始化；权重与偏置均可训练',14)
    arrow(ax,40.5,90.5,40.5,94)
    box(ax,10,94,61,9)
    text(ax,40.5,98.5,r'完整输出  $O\in\mathbb{R}^{B\times T\times2}$',18,ha='center')

    tag(ax,85,32,1,'线性层作用于最后一个维度')
    text(ax,88,39,r'$W_c\in\mathbb{R}^{2\times768},\quad b_c\in\mathbb{R}^{2}$',22)
    text(ax,88,47,r'$O=H W_c^{\mathsf{T}}+b_c$',25)
    text(ax,88,53,'转置后：768 × 2；偏置沿 B、T 广播',15)
    text(ax,88,58,'线性层参数量：2 × 768 + 2 = 1538',15)
    tag(ax,85,66,2,'只取最后一个位置的输出')
    text(ax,88,72,'logits = model(X)[:, -1, :]',16,font=MONO)
    text(ax,88,78,r'$Z=O_{:,-1,:}\in\mathbb{R}^{B\times2}$',22)
    text(ax,88,84,'索引 -1 选择序列维度；Batch 维度保留',14)
    tag(ax,85,92,3,'输出类别')
    text(ax,88,99,r'$\hat y_b=\arg\max_{c\in\{0,1\}} Z_{b,c}$',22)
    text(ax,88,104,'argmax / softmax：无可学习参数',13,color=GRAY)
    text(ax,7,112,'注意：脚本读取填充后的最后一个位置，并未按每条短信的真实长度定位末词元。',14)

    # Figure 3: training and evaluation; exact code timing and aggregation.
    f,ax=page('GPT-2 短信分类：训练循环与评估调用',
              'calc_loss_batch 计算分类损失；train_classifier_simple 驱动参数更新',3)
    panel(ax,6,21,72,87,'一、每个训练批次的计算')
    panel(ax,82,21,72,87,'二、评估时机与指标统计')
    box(ax,10,30,64,10)
    text(ax,13,33.5,r'输入 $X:[B,T]$；类别标签 $y:[B]$',17)
    text(ax,13,37.5,'optimizer.zero_grad()；训练 B = 8',14,font=MONO)
    arrow(ax,42,40.5,42,44)
    box(ax,10,44,64,11)
    text(ax,13,47.5,'logits = model(X)[:, -1, :]',15,font=MONO)
    text(ax,13,52,r'类别分数 $Z:[B,2]$ → cross_entropy',16)
    arrow(ax,42,55.5,42,59)
    box(ax,10,59,64,19)
    text(ax,13,63,'分类交叉熵（返回一个标量）',16,bold=True)
    text(ax,42,71,r'$\mathcal{L}=-\frac{1}{B}\sum_{b=1}^{B}\log\frac{\exp Z_{b,y_b}}{\sum_{c=0}^{1}\exp Z_{b,c}}$',23,ha='center')
    arrow(ax,42,78.5,42,82)
    box(ax,10,82,64,19)
    text(ax,13,86,'loss.backward()：反向传播',15,font=MONO)
    text(ax,13,91,'optimizer.step()：更新参数',15,font=MONO)
    text(ax,13,97,'仅更新可训练参数；冻结参数不更新',14)
    text(ax,10,104,'直接传入 logits；计算损失前无需 softmax。',13,color=GRAY)

    tag(ax,86,32,1,'每 50 步记录损失（含第 0 步）')
    box(ax,86,37,64,21)
    text(ax,89,41,'evaluate_model',16,font=MONO)
    text(ax,89,46,'→ calc_loss_loader → calc_loss_batch',13,font=MONO)
    text(ax,89,51,'训练 / 验证各取至多 5 个批次',15)
    text(ax,89,55.5,'eval() + no_grad()；结束后恢复 train()',13,font=MONO)
    tag(ax,86,65,2,'每轮结束记录准确率')
    text(ax,89,71,'calc_accuracy_loader：至多 5 个批次',13,font=MONO)
    text(ax,89,77,r'$\mathrm{Accuracy}=\frac{N_{\mathrm{correct}}}{N_{\mathrm{eval}}}$',21)
    text(ax,89,82,'正确预测样本数 ÷ 已评估样本数',13,color=GRAY)
    tag(ax,86,87,3,'训练完成后计算最终准确率')
    text(ax,89,93,'遍历训练 / 验证 / 测试 loader 的全部批次',14)
    text(ax,89,98,'训练 loader 仍丢弃不足 8 条的末批',14)
    text(ax,89,103,'损失按批次等权平均；准确率按样本统计',13,color=GRAY)
    text(ax,7,112,r'AdamW：$\mathrm{lr}=5\times10^{-5}$，weight_decay = 0.1；共 5 轮；examples_seen 累加样本数。',14)

    with PdfPages(ROOT/'GPT2_短信分类_三图.pdf') as pdf:
        for i,f in enumerate(figures,1):
            f.savefig(ROOT/f'{i:02d}_figure.png',dpi=150,facecolor='white')
            f.savefig(ROOT/f'{i:02d}_figure.svg',facecolor='white')
            pdf.savefig(f,facecolor='white')
            plt.close(f)

    (ROOT/'README.md').write_text('''# GPT-2 短信分类微调技术示意图

    依据附件 gpt2_spam_classifier_organized.py 绘制。图中不含推测的运行结果。

    1. 数据集与 DataLoader：编码、统一长度、单条取样、自动组批。
    2. 分类读出与微调范围：分类头矩阵维度、偏置广播、最后位置选择。
    3. 训练与评估：分类交叉熵、参数更新、评估调用与聚合方式。

    每张 PNG 为 2400 × 1800，附矢量 SVG、三页 PDF 和可修改的绘图脚本。
    中文使用 Noto Serif CJK SC Regular / Bold；英文衬线采用 Nimbus Roman
    （环境无 TeX Gyre Termes）；代码使用 Latin Modern Mono，数学采用 Computer Modern。
    SVG 字形转为路径以保持跨设备外观；内容修改建议使用 draw_figures.py 重新生成。

    范围与代码细节：
    - chapter04.py 未提供，图中 GPT 内部未展开，未推测其内部实现。
    - 文本列为小写 text，标签列为 Label；脚本未给出类别 0/1 的文字映射。
    - T 为训练集最长编码长度，实际值未知；代码未验证 T <= 1024。
    - 分类取填充后序列的最后位置，而非按真实文本长度定位末词元。
    - 训练 drop_last=True；最终训练准确率仅覆盖训练 loader 实际返回的样本。
    - 损失按批次平均；准确率按样本统计；两者的权重规则不同。
    - 第 0 步评估发生在第一次参数更新之后；随后在第 50、100……步评估。

    运行：python draw_figures.py（需要 NumPy、Matplotlib、fontTools 及脚本指定的字体）。
    ''',encoding='utf-8')
    with zipfile.ZipFile(ROOT/'GPT2_短信分类_图片与源文件.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(ROOT.iterdir()):
            if p.suffix in {'.png','.svg','.pdf','.py','.md'}:z.write(p,p.name)
    print('Created 3 PNGs, 3 SVGs, a PDF and a ZIP.')


if __name__ == "__main__":
    main()
