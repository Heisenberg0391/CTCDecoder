# coding=utf-8
from __future__ import division
from __future__ import print_function
import numpy as np


class BeamEntry:
    "information about one single beam at specific time-step"
    def __init__(self):
        self.prTotal = 0 # blank and non-blank
        self.prNonBlank = 0 # non-blank
        self.prBlank = 0 # blank
        self.prText = 1 # LM score
        self.lmApplied = False # flag if LM was already applied to this beam
        self.labeling = () # beam-labeling


class BeamState:
    "information about the beams at specific time-step"
    def __init__(self):
        self.entries = {}

    def norm(self):
        "length-normalise LM score"
        for (k, _) in self.entries.items():
            labelingLen = len(self.entries[k].labeling)
            self.entries[k].prText = self.entries[k].prText ** (1.0 / (labelingLen if labelingLen else 1.0))

    def sort(self):
        "return beam-labelings, sorted by probability"
        beams = [v for (_, v) in self.entries.items()]
        sortedBeams = sorted(beams, reverse=True, key=lambda x: x.prTotal*x.prText)
        return [x.labeling for x in sortedBeams]


def applyLM(parentBeam, childBeam, classes, lm):
    "calculate LM score of child beam by taking score from parent beam and bigram probability of last two chars"
    if lm and not childBeam.lmApplied:
        c1 = classes[parentBeam.labeling[-1] if parentBeam.labeling else classes.index(' ')] # first char
        c2 = classes[childBeam.labeling[-1]] # second char
        lmFactor = 0.01 # influence of language model
        bigramProb = lm.getCharBigram(c1, c2) ** lmFactor # probability of seeing first and second char next to each other
        childBeam.prText = parentBeam.prText * bigramProb # probability of char sequence
        childBeam.lmApplied = True # only apply LM once per beam entry


def addBeam(beamState, labeling):
    "add beam if it does not yet exist"
    if labeling not in beamState.entries:
        beamState.entries[labeling] = BeamEntry()


def ctcBeamSearch(mat, classes, lm, beamWidth=10):
    "beam search as described by the paper of Hwang et al. and the paper of Graves et al."

    blankIdx = len(classes)
    maxT, maxC = mat.shape

    # initialise beam state
    last = BeamState()
    labeling = ()
    last.entries[labeling] = BeamEntry()
    last.entries[labeling].prBlank = 1
    last.entries[labeling].prTotal = 1

    # go over all time-steps
    for t in range(maxT):
        curr = BeamState()

        # 每个timestep保留前beamWidth个top概率标注
        bestLabelings = last.sort()[0:beamWidth]

        # go over best beams
        for labeling in bestLabelings:

            # probability of paths ending with a non-blank
            prNonBlank = 0
            # in case of non-empty beam
            if labeling:
                # probability of paths with repeated last char at the end
                prNonBlank = last.entries[labeling].prNonBlank * mat[t, labeling[-1]]

            # probability of paths ending with a blank
            prBlank = (last.entries[labeling].prTotal) * mat[t, blankIdx]

            # add beam at current time-step if needed
            addBeam(curr, labeling)

            # fill in data
            curr.entries[labeling].labeling = labeling
            curr.entries[labeling].prNonBlank += prNonBlank
            curr.entries[labeling].prBlank += prBlank
            curr.entries[labeling].prTotal += prBlank + prNonBlank
            curr.entries[labeling].prText = last.entries[labeling].prText # beam-labeling not changed, therefore also LM score unchanged from
            curr.entries[labeling].lmApplied = True # LM already applied at previous time-step for this beam-labeling

            # extend current beam-labeling
            for c in range(maxC - 1):
                # add new char to current beam-labeling
                newLabeling = labeling + (c,)

                # if new labeling contains duplicate char at the end, only consider paths ending with a blank
                if labeling and labeling[-1] == c:
                    prNonBlank = mat[t, c] * last.entries[labeling].prBlank
                else:
                    prNonBlank = mat[t, c] * last.entries[labeling].prTotal

                # add beam at current time-step if needed
                addBeam(curr, newLabeling)

                # fill in data
                curr.entries[newLabeling].labeling = newLabeling
                curr.entries[newLabeling].prNonBlank += prNonBlank
                curr.entries[newLabeling].prTotal += prNonBlank

                # apply LM
                # applyLM(curr.entries[labeling], curr.entries[newLabeling], classes, lm)

        # set new beam state
        last = curr

    # normalise LM scores according to beam-labeling-length
    last.norm()

    # sort by probability
    bestLabeling = last.sort()[0] # get most probable labeling

    # map labels to chars
    res = ''
    for l in bestLabeling:
        res += classes[l]

    return res

def beam_search_decoder(data, k):
    sequences = [[list(), 1.0]]
    # walk over each step in sequence
    for row in data:
        all_candidates = list()
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                candidate = [seq + [j], score * -np.log(row[j])]
                all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup:tup[1])
        # select k best
        sequences = ordered[:k]
    return sequences

def testBeamSearch():
    "test decoder"
    classes = "，。、：0123456789的一是不了在有人这上大来和我个中地为他生要们以到国时就出说会也子学发着对作能可于成用过动主下而年分得家种里多经自现同后产方工行面那小所起去之都然理进体还定实如么物法你好性民从天化等力本长心把部义样事看业当因高十开些社前又它水其没想意三只重点与使但度由道全制明相两情外间二关活正合者形应头无量表象气文展系代加各很教新向她机内此老变原结问手日利质已最政儿见并平资比特果什建反常知第电思立提或通解身四品几位别论公给少条观回海基次被山才己期西术济认先命走真员及数话门级军统光声题入美口感战科程式指世必放打总接做东区农强造类受场五直月流决干则更色处路运任具目再治神求件管组根阶将改导眼规识革计白马金界取市语设完究党女传风信名便保育队带叫研领北较张即至许步往听调务花争线呢每边难太共交确劳据达住收候需转百南清格影书切且却志热联安极今单商料技深验增记近言整精集空连报觉车价音响存办怎病快图况例消容史非离节亲万八构族石满何广律青林克王历权素始断九际积吃艺态证众创红望须群师该复细包土持服笑德般远爱准写算火死半布随六元低称引照失养视习段字织斗团器兴乐效显斯千落示仅企似备除支标早吧周速跟七采状吗约城层专划轻拉值适英告讲维营士环古院让按型足势毛台紧片属严树画厂功注演源温某找参易飞推围列客河虽境食李尔黄黑念越选华角考致夫初装议首委底另江息压密孩村费局派房星突供曾排苦够像站校富谈米充阳破母球射批配纪未哪妈差刚率察舞依占呀微益础刻倒举血故互范木号尽请脸兵职留铁吸钱急独剧查皮激坐乎固害夜续京双拿府限草项述曲既春官坚令句助简杂材封护司省县模试绝洋谁继止喜优词陈鱼终施晚朝含仍脑甚汉修油衣击云送巴普征错均投余波友武责游敌叶船久否异块脚怕田策苏乡帝靠医测植啊括换逐娘罗戏善获待降冲父置左右沉酸亮良班著亚抗啦静渐概居跑旧防买挥负帮欢冷液停减男胜答析忙短烈映财背奇药承岁略穿牛室移饭坏假州缺愿练超雨慢竟呼散灵副歌谓虫读介培卫宗章伤评丰核预担诉惊刘景乱氧阿龙宣升爷洲判案协福货味若促希握端针屋掌痛检闻妇贵顾沙扩困嘴哥销印免烟跳香纸户板睛误探座盾灯搞祖顺婚赶季兰忽践熟矛银爸临佛犯胞讨陆补圆童归弟域索罪脱追守课顶松杀掉野训遍庭弱赛额酒绿席露卖钟旁央肉染欧怀稳伯编抓玉咱唯唱释翻控怪阵烧堂杨您诗灭夏敢退姑恶透鲜附骨审胡输宁丝姐彩睡巨迫伙鼓借耳典刺暗镜序哈菜溶笔汽礼摇迅险遗沿威弹私攻宝抽雪孙毒藏顿末摆趣救镇楼智园永康遇雄替危楚侵枪冰健播途损秀丽执秋束纳疑午虑粮架盛湖尚床份套毫奋扬赵混献择庄殖店皇购缩盐冬齐径偏盖款衡阻托喊繁牙纯殊浪招符折泽弄奶谢累亦厚朋伸筑闪枝授蛋潮登售孔街败迎肥恩尼幼惯钢肯矿岛伟粉振吴尤夺忘荣阴亡凡毕暴唐勇避麻硬股洞盘轮闹警绍描录寻妹幸壁距休玩延综甲泛哲载雷窗虚俗秘倾哭禁零触巧圈努喝彻晶丁操奴宜菌叔桥仿默航泪谷予挂壮纷粒燃诸削婆苗付迷馆灰鸡岸寒泥梅残税虎庆抱拍鸟墙缓横尾督械莫尖旅倍措剥遭懂粗奏润侧辩茶鬼梦贯篇妻宽腿桌刀浓跃蹈洗卷欲袋赏劲悲赞氏鲁迹汇竞醒裂森闭腰磁湿聚兄徐炼订川蒙吹胸召伴徒幅疗淡奖珠仪绪蒋猪剂访狗摸趋宋伏棉恐羊碳隔网猛爬缘麦库津监薄净旋贫罢穷乏挑废耗汗尊牌抢软隶宇圣港欣塞刑博揭垂姓荒诚捕炮呈辑卡伦蓝抬宫敬版忍颜竹碰启隐票脉抵浮煤剩珍奔堆扎谋貌脏逃洪旗潜鼻忆湾绩恰坦怒震豆戴插瓦涉绘龄池沟呵伍氢拥乘吉朱蒸舍尺贡梁糊漫腾症丈钻磨井艰键绕幕斤污贝冒阔胆渡碍愈崇岩玻碎宪俄障融贸耐届寄荡卵俩币弃悄纵陷偶坡勤享塑膜铜拖暖稍摩硫哩乌循妙疾肩炎偷灾赤恒仁腔孤杯乃稀恢违伊挺乙拜焦锋铺颗滑敏洁凝嘛凉杜凭鞋勒递爆迁秦桂璃摄叹峰牧恨糖骂紫贴壤仰辛腐割辈君泡郭呆番励亿乳兼昨刊擦迟拔丹彼眉搬滴朵遵畜逻患奥忠耕鉴挤滚壳厅旦炸塔弯频狂陶晓柳炉宏愤瞧箱扑暂盆瓶柱辉雅衰洛朗琴腹牲黎雾肃冠伪幻仔杰烦悟荷晨甘饱飘芳肌舒惜丧肚灌抑鸣辞宙邻昌扫赖泉援鼠肠墨啥祥储萨裁奉傅姿埋亩犹咬瓜粘芽吐租昆估碗轨稻役纲烂昏毁勃饰隆曹仙夹爹郑闲柔吓嫂巩亏浅帽斜详佳饮猫杆恋凤缝疏扶喷厉截扰拨雕瑞码讯莲邦贷丢慌寿毅涂屈漂蜂猎氯劝柴悉狠饲岗祝皆罚桃躺猴宿胶锅偿译宾袭串申捉挖窝踏邓允舌溜驻膀渔纠冻扭臣埃辐涨闷签耀秒贺辨燕慰颇债肺逼魂匆旺姻吞甜吨掩屏搭涌袁汤岭拟慧郎瘦阅辆袖填匀娃筒豪籍轰餐尘敲帐撞钠牢肿盲腊锐骗返寺骑扣醉沈纹纺泰辟盟丛忧拒魏躲肖纤悬晋旱誉谱祭狭箭饿驱脂冯斥肤椅仗栽墓兽舅碱锁喂抛苍臂溪胖汪伐刷稿绳剪唤凶邮齿尝漠怨络撒撤筹幽渗蛇诱抚磷逆牵诊株庙抖惨浑驶泼厌怜疲叙厘跨嫁胃惠哀梯蜜诺悠滩添笼陪挨拳寂晃催倡朴郁脊扇魔仓蓄哎棵胀眠劣秩塘媳孟廷胎铃抹吵肢拾寨聪翠赫尿疼览狼轴锦夸翼牺剑卧姆曼匠熊搜盈覆唇驾腺辅枯鞭喘翅肝掘歪欺疆赋婶畅葡脾哼柜兔顽藻鸭睁侦页谐仇滋弦锡炭拼茫锻脆卢裤萌骤悦葬氮渠诞棒斑浩遥盒颤扮玲颈扯摊丑桑盗戒柏唉羽凌胁皱捷堡挣嚷陵淋彭茂吊愁巡纱惑寸董勾阐昂纬逢披恼佩妥萄姨坛署肾迈仲邀辽莱阀弥猜瞪韩臭脖掏妨燥酷蚁鸿棚芦漏凯浸舰狱踪摘氛御蔡硝扔裕竭氨愉坑疯蚕嘉矮泄掠叉冶渴囊辣菊凑歇嘻雇婴泌痕夕亭跌贮螺册贪庞淀砖铝耶艳涛僚玛绵拆遮贤慎贾糟鹿霸匹咳挡舱雀赴蚀鹰咽饥询嘿函堪嗓晴嫩豫辖蹲酬剖钾尸胚哦悔雌漆惩链篮筋遂摔叠垄霍侯匪贞巾丘滞洒蓬纽浙潘娜矩盼闯叛霞履瞬砍奸菲浆敦拓僧儒棋芬慈俺萧蛮郊勉钉缠吾咕盯膨堵卜扁哟寡葛鄂驰桶晒慕谊霜栏哇隙畴押孝竖韵旬衫欠吕喉霉孕邪峡穆硅羞湘澳慨惧穴辱捞炕棍撑窄宅崖陕俱晰爽鹅疫绒巢苹茎捧忌搏焰惹躯奈岂甫拦缚趁艾橡祸碧卑攀饼蔬譬灶曰赔粪颠媒蛙殿逝钩衬罩哑艇岳裸庸嫌砂窃涵熔遣梨泊廉傲傻畏缴钙乔铅萍懒莉挽凄卓刮妄劫档掀藤哗阁吟杏颂梢罕沸滨汁翔戈愚秧抒姜弊屁框拘贩嘱喻垫迪契瑶弗赌堤哄旨晕凳址彪贼韦踢俊稚账蝶帕凹携茅枢哨枚溃歧驳弓抄稼徽颁杠泳蜡沾绣吻谨叮厨拐衷宴鸦戚粹碑煮淑蕾逸翁倘恳逊趟衍讼蔽罐胳裙龟奠驴颌撕姚滥恭蒂蕴骄耻煌冤瞎娇杭愧倦淮柯聊侨惟萝谅蚂瘤崔铸廊溉髓沫碌拢娱帅吏坟昼歉峻刹坝渊栖捐傍鸽叭呜淘轿宰玄谦丙朽愣跪寓券捏搅槽笨爪咋厦奎榜驼屠耸赚殷甩僵歼逮孢矣锤兹浴呐坎珊苯垦芝吁兜禅舆酶惶咐颖陌瓷赢帘裹摧兆卸瞅噢咧仆饶醇膊膝篷肪嗯寞挫蚊桔俯眨巷熙烛缸踩屯顷鼎砸咸乾衔沃俘熬桩侍禽芒喇钓灿凸泣吼讽谭聘敞哺浦癌躁卿勘舟溢楞窜蛛虹烷耍饵枣骚浇崩蹦褐嘎荫仑窟俭陡钦倩嚼弧涡佐夷潭眯廓坊缔枕蹄捣猩狐柄穗伞拚浊壶搁冈屑谬拱斋荧祀誓魁姊虾魄娟绸泻秃鹏蟹斌灼镁扛瓣艘捆膏厢苇蒲妮啸劈滤葱袍栗耽颊痴猿沪絮怯蔑垒咯芭笛瞒爵冀嘲酿赐惕烫挪逗吱淫糕渣赠塌琼剿拣澄暑媚荐鹤凿讶烁狮绅乒阎薛侮嗽喃朦埔蝇娶锣炒棕陀肆韧禄礁乖驯萎匙骆昔滔摹卤舶雁裳陋耿妆崭勋闸侣喧膛痰胧辫贬烯缀幢铀窑丫羡坪犁坤甸淹汹柑杉薯腻棺怖锈洽魅炳椎诵畔咪玫帜桐柿襟筷帆匈唔佣梳嵌鳞噪熄莎砌昧簧婉毯憾孵谎坠嗡煞藉绑妖苔揉蝴瑰昭莹拌汰拂撰嗅钮鄙盏谣稣烘皖竿锥烃犬皂蕊蔓喀疮咒僻薪铭寇炯恍辰祈迭斩屡豁啼烤椒稽啡籽澡洼茨巫唾榴乞焊蜘杖隋晌暮贱咚扒涤拴闽茧伽叨喳怔腕咖斧啄溯踱脓沦畸锄擅疙炊榨粤莺绎尉撇虏棱簇豹涩乓柬憎涯蔚狩茄觅挚姥捡袄镶袜旷碘丸糙倚锌硕矢牡鲸褂铵淌憨禾尹潇勿嗦梭庐聋荆煎苟醋秉彬掷茵蓉噜悼矫琢翘挠吭憋蔗痒瑟兑璋琐酋拧廖槐闺彰迄呕搂菱筐焚瘩钥谜缕垮嘟蠢咨馈蛤聂缅敷宠沼榆禧巍宛俞眶绷筛菩吩钞诈酱葫嗨佃颓渺揽焕躬歹沛啪溅栋砚霖侄囚氓狡妃沁峭瞥揣禹樱檐暇澜岔粥涅馏嗜麟笃铲寅尧剔屿赣扼稠胺帖揪亥堕懈梧窍瀑麓卒淤腥腮鹃绞蕉枫脯腑熏溴薇婷莽磕酵碟毙挟棘庚睹弛秤蚓枉茸俏锰葵咀逛梗祷炽绰拙婿蜀钧狄掺鞍嚣紊蚯圃垃蛾韬笋啃鲫酚隘哮诬鲤叽惰嘀毡勺羔湍瘫狸郡娥钝趾靖跋湛咦畲咙锯圾弘袱妒鲨翩颅嗬忿藩弼捂刨虐灸芙痹鹊绢簿诏棠徊蹬戳澈沧扳疤佑匣湃疟烙窥慷眷窖呱淳焉轩芸琳恺嘶馒哆绥捎崎徘冥嫉姬辜诀窘逾裴诡龚峨橙诧轧咏衙椭豚粟嗣虔讳梵吆邢剃奢叩梆邹瑛淆芹拎缭厥翰斡髦芯驮羌诫卉郝骼磅崽陨蜕屎敛迦桦墩鳃倪讥氟宵晤趴寥苛瑚祠惭靶肘眺钳砾倌厕拽酝怡醛镰盔菇搓甭呗鲍涕蒜噬愕囱鸥唠壕孚濒呻拇撩琅撮渤胰嘘襄佟皿鞠桨蝗泓酌伶倔蝠橱酯圳绚睦秸窦庇黯窒兀屉庶赎戎沥妓樊侗滇岱簸迸苑眩旭骡凛撼骇雏懊庵栅汝喽峪瘾徙蠕冉邱蜻挎茬萤廿荔癞亢秽嚎篡猖绊攘呛瞄汞鳄贻寝黔啤募挛漉陇兢汲遏瞻墟惋酥煽篱婪橘邯氦阜缎刁嗒蝙狈褶肋岚尴尬舵潺埠谕蜓亨苷於裔蓦贿悍逞砧迂嬉邵刃谛陛啰粱揩猾苞斟黝芜酮赦沐纶坷馋孜哉彦捅钛擒烹琶涝缆傣怅蝉缤泵虞雍栓砰痉胱锭嘈辙钊札漾霎瓢匿楠蟆磺卦瞩疹疡臀缫溺蘑螂捍跺汛膳瑁骏皓玳嫦瘪璧惫肴惘伺峦糠疚涧浒呃搀睬拯曳蜒殉涎囤祺澎牟靳凰怠娴戊佤鳍窿捶踌鬓蔼瘀拭悯惚厄躇峙驭彝抡鲢剌晦秆沌哒槌邑馨瞳簌塾阮殴靡踞寰吠丞钡奕褪檀琵漓侃攫镀栈靴酰毋褥茹桓哧乍墅删碾撵眸螨驹碉雹嗳啧炫捻痪沽黛渍赘舷蜗恕谴噗宦唬惦憧悖肇辗翌腋荀曙拗幔睫斐郸忱鳖蚌隧厮锹耦酣蝎饷蜿邃朔狞苓伎蕃垛詹椰掖栩戌鑫迢瞟谍渲鞘孽篓禀揍竺绽垣芋偎唰岖跄粕袅抿瀚萃矗盎俨遐瘟汀蟀裘缦踉晾缉岑菠攥藓瓮恬搔岐坯咆糯莓娄肛嚓蘸柚龈屹蟋犀搪炔瑙珑唧噶蹭臻鹦臆喔嗖倏懦捺懵珞犊葆颐衅偕腆坍笠擂楷翡叼侈帚侠舔曝喏镍憬骥垢筏嗤肮诣夭惺箫鹉榻舜穹掐燎竣蕨崛沮臼鸠芥壑凋殆箔炙楂霄僮荪榕琦祁蹑磊鞑婢赃踊蛹娅祟笙妊吝矜娣跛攒韫奚帷瞿腼捋饺羁荤锢蟾蛀悻抨哝馍糜粼槛婕杈樟漩兮蜷擎赁蛎炬髻鼾铬隅伢曦焘仕姗佧佼跤缰悸诙隍槟胥蜥妩娓淞恃琪猕锚迥匾镑梓渝螳娠芍妾癖螃辘痢阑瘠箕搐骋忏粽篆拄铮翕汐渭豌甥纫耙撂哽咄褒甄驿沓嵩汕睿憔霓箩瞌膈镐巅忖悴忡掰踵臧媛奄桅吮祛锵琉瑾钴篝篙丐萼纂谚钨滕擞熵毗浏彗遁耷磋蹙淙叱馅涮虱冗蟒枷盹敝惴萦薰彤嚏赡跷藕蘖鳗埂嗔摒诅伫祯缨玺蛐怦谟冕儡蝌窠泾暨亘腭蚪钵锗耘佯殃邬啕讪跚仨蹂觑憩匍泯娩楣镭傀匡晖诃匐掸坳咂熠侥筱唏漱畦勐瞭獗犷颚桉盅筝囿噎黍慑闰蔷蹒寐佬谏幌貂癫咿蚜圩袒酞唆昵羹麸蜴恤潼惬咛蔫聆蔺俐榄撅扈梏沣蹿谤鸳瑜唷腌撬痞柠瞰绛橄痘掂嫡椿镖匕匝漪楔肽噙镊檬躏噘厩矾惆浜皑冢桎蚴垠鸯蹋羟硼缪赂抠铿赳睑徨蒿掇褚掣粑镯茁膺煦螅痊蹼囔萘翎匮铐咔蛔涸辄阈晔堰韶卅沏枇杷颞谧醚龛韭皎麂矶诽炖僰铂桢瘴舀怏鞅锨忐揿樵绮妞茉蛟岌阂圭臌睾帛酉臃铣坞俟嫔彷螟闵嶙啷斓骸刽绯俾鬃栉狰朕蚤沂蓿逍倭忑飒瘸脐溥霹拈牦掳褴逵咎蓟渎邸爨猝怆礴笆釉勰淅濡噼疱啮嘹涟锂俚抉亟汾惮蹴螯蚱恽晏挝腧缮啬臊堑苜淼榈冽雳烬戟馄烽讹啶喋刍裨鳌谙翟毓沱饨燧戍偌嚅颧媲绫潦轲戛荚蝈蜚蜍坂漳纭俑胭悚珀泞濛懑褛菁嗥噩铎娼殡敖亵硒孰舢榷蹊弩嗫滦鹭镂寮夔颏阱粳腱涣竑蚶扉幡啜敕翱呓"
    mat = np.loadtxt('softmax_mat.txt',dtype=np.float64)
    # mat = np.array([[0.4, 0, 0.6], [0.4, 0, 0.6]])
    print('Test beam search')
    expected = '谁烧罚谁'
    actual = ctcBeamSearch(mat, classes, None)
    # actual = beam_search_decoder(mat, 3)
    print('Expected: "' + expected + '"')
    print('Actual: "' + actual + '"')
    print('OK' if expected == actual else 'ERROR')


if __name__ == '__main__':
    testBeamSearch()
