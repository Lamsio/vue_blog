(window.webpackJsonp=window.webpackJsonp||[]).push([[59],{388:function(t,s,a){"use strict";a.r(s);var n=a(4),e=Object(n.a)({},(function(){var t=this,s=t._self._c;return s("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[s("div",{staticClass:"language-python line-numbers-mode"},[s("pre",{pre:!0,attrs:{class:"language-python"}},[s("code",[s("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("import")]),t._v(" numpy "),s("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("as")]),t._v(" np\n"),s("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("import")]),t._v(" pandas "),s("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("as")]),t._v(" pd\n"),s("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("import")]),t._v(" tensorflow "),s("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("as")]),t._v(" tf\n"),s("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("import")]),t._v(" matplotlib"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("pyplot "),s("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("as")]),t._v(" plt\n")])]),t._v(" "),s("div",{staticClass:"line-numbers-wrapper"},[s("span",{staticClass:"line-number"},[t._v("1")]),s("br"),s("span",{staticClass:"line-number"},[t._v("2")]),s("br"),s("span",{staticClass:"line-number"},[t._v("3")]),s("br"),s("span",{staticClass:"line-number"},[t._v("4")]),s("br")])]),s("h4",{attrs:{id:"sigmoid"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#sigmoid"}},[t._v("#")]),t._v(" Sigmoid")]),t._v(" "),s("p",[t._v("常用于二分类问题上，公式为："),s("strong",[t._v("$$\\sigma(x) = \\frac{1}{1+e^{-x}}$$")])]),t._v(" "),s("div",{staticClass:"language-python line-numbers-mode"},[s("pre",{pre:!0,attrs:{class:"language-python"}},[s("code",[s("span",{pre:!0,attrs:{class:"token comment"}},[t._v("# Sigmoid函数 (输出层)")]),t._v("\n"),s("span",{pre:!0,attrs:{class:"token comment"}},[t._v("# 定义x的取值范围")]),t._v("\nx "),s("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" np"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("linspace"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),s("span",{pre:!0,attrs:{class:"token operator"}},[t._v("-")]),s("span",{pre:!0,attrs:{class:"token number"}},[t._v("10")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),s("span",{pre:!0,attrs:{class:"token number"}},[t._v("10")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),s("span",{pre:!0,attrs:{class:"token number"}},[t._v("100")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n"),s("span",{pre:!0,attrs:{class:"token comment"}},[t._v("# 直接使用tensorflow实现")]),t._v("\ny "),s("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" tf"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("nn"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("sigmoid"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("x"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\nplt"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("plot"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("x"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v("y"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\nplt"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("grid"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n")])]),t._v(" "),s("div",{staticClass:"line-numbers-wrapper"},[s("span",{staticClass:"line-number"},[t._v("1")]),s("br"),s("span",{staticClass:"line-number"},[t._v("2")]),s("br"),s("span",{staticClass:"line-number"},[t._v("3")]),s("br"),s("span",{staticClass:"line-number"},[t._v("4")]),s("br"),s("span",{staticClass:"line-number"},[t._v("5")]),s("br"),s("span",{staticClass:"line-number"},[t._v("6")]),s("br"),s("span",{staticClass:"line-number"},[t._v("7")]),s("br")])]),s("h4",{attrs:{id:"tanh"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#tanh"}},[t._v("#")]),t._v(" Tanh")]),t._v(" "),s("p",[t._v("数学中的双曲正切函数Tanh也是一种神经网络常用的激活函数，尤其是用于图像生成任务的最后一层，其计算方式为："),s("strong",[t._v("$$f\\left(x\\right) = \\frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}$$")])]),t._v(" "),s("p",[s("img",{attrs:{src:"/more/Pasted%20image%2020230718205039.png",alt:"avatar"}})]),t._v(" "),s("div",{staticClass:"language-python line-numbers-mode"},[s("pre",{pre:!0,attrs:{class:"language-python"}},[s("code",[s("span",{pre:!0,attrs:{class:"token comment"}},[t._v("# Tanh函数 (隐藏层)")]),t._v("\n"),s("span",{pre:!0,attrs:{class:"token comment"}},[t._v("# 定义x的取值范围")]),t._v("\nx "),s("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" np"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("linspace"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),s("span",{pre:!0,attrs:{class:"token operator"}},[t._v("-")]),s("span",{pre:!0,attrs:{class:"token number"}},[t._v("10")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),s("span",{pre:!0,attrs:{class:"token number"}},[t._v("10")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),s("span",{pre:!0,attrs:{class:"token number"}},[t._v("100")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n\ny "),s("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" tf"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("nn"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("tanh"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("x"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n\nplt"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("plot"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("x"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v("y"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\nplt"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("grid"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n")])]),t._v(" "),s("div",{staticClass:"line-numbers-wrapper"},[s("span",{staticClass:"line-number"},[t._v("1")]),s("br"),s("span",{staticClass:"line-number"},[t._v("2")]),s("br"),s("span",{staticClass:"line-number"},[t._v("3")]),s("br"),s("span",{staticClass:"line-number"},[t._v("4")]),s("br"),s("span",{staticClass:"line-number"},[t._v("5")]),s("br"),s("span",{staticClass:"line-number"},[t._v("6")]),s("br"),s("span",{staticClass:"line-number"},[t._v("7")]),s("br"),s("span",{staticClass:"line-number"},[t._v("8")]),s("br")])]),s("h4",{attrs:{id:"relu"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#relu"}},[t._v("#")]),t._v(" ReLU")]),t._v(" "),s("p",[t._v("ReLU激活函数可以说是深度神经网络中最常用的激活函数之一，常用于隐藏层，计算方式："),s("strong",[t._v("$$f\\left(x\\right) = \\max\\left(0, x\\right)$$")])]),t._v(" "),s("p",[s("img",{attrs:{src:"/more/Pasted%20image%2020230718205014.png",alt:"avatar"}})]),t._v(" "),s("div",{staticClass:"language-python line-numbers-mode"},[s("pre",{pre:!0,attrs:{class:"language-python"}},[s("code",[s("span",{pre:!0,attrs:{class:"token comment"}},[t._v("# ReLu函数 (隐藏层)")]),t._v("\n"),s("span",{pre:!0,attrs:{class:"token comment"}},[t._v("# 定义x的取值范围")]),t._v("\nx "),s("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" np"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("linspace"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),s("span",{pre:!0,attrs:{class:"token operator"}},[t._v("-")]),s("span",{pre:!0,attrs:{class:"token number"}},[t._v("10")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),s("span",{pre:!0,attrs:{class:"token number"}},[t._v("10")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),s("span",{pre:!0,attrs:{class:"token number"}},[t._v("100")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n\ny "),s("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" tf"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("nn"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("relu"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("x"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n\nplt"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("plot"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("x"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v("y"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\nplt"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("grid"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n")])]),t._v(" "),s("div",{staticClass:"line-numbers-wrapper"},[s("span",{staticClass:"line-number"},[t._v("1")]),s("br"),s("span",{staticClass:"line-number"},[t._v("2")]),s("br"),s("span",{staticClass:"line-number"},[t._v("3")]),s("br"),s("span",{staticClass:"line-number"},[t._v("4")]),s("br"),s("span",{staticClass:"line-number"},[t._v("5")]),s("br"),s("span",{staticClass:"line-number"},[t._v("6")]),s("br"),s("span",{staticClass:"line-number"},[t._v("7")]),s("br"),s("span",{staticClass:"line-number"},[t._v("8")]),s("br")])]),s("h4",{attrs:{id:"softmax"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#softmax"}},[t._v("#")]),t._v(" Softmax")]),t._v(" "),s("p",[t._v("一种用于多分类问题的激活函数，在多类分类问题中，超过两个类标签则需要类成员关系。对于长度为 K 的任意实向量，Softmax 可以将其压缩为长度为 K，值在（0，1）范围内，并且向量中元素的总和为 1 的实向量。")]),t._v(" "),s("p",[s("img",{attrs:{src:"/more/Pasted%20image%2020230718205949.png",alt:"avatar"}})]),t._v(" "),s("h4",{attrs:{id:"如何选择激活函数"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#如何选择激活函数"}},[t._v("#")]),t._v(" 如何选择激活函数")]),t._v(" "),s("h6",{attrs:{id:"隐藏层"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#隐藏层"}},[t._v("#")]),t._v(" 隐藏层")]),t._v(" "),s("ul",[s("li",[t._v("优先选择RELU函数")]),t._v(" "),s("li",[t._v("如果RELU效果不好，可以尝试Leaky ReLu")]),t._v(" "),s("li",[t._v("不要使用sigmoid激活函数，可以尝试使用tanh函数")])]),t._v(" "),s("h6",{attrs:{id:"输出层"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#输出层"}},[t._v("#")]),t._v(" 输出层")]),t._v(" "),s("ul",[s("li",[t._v("二分类问题选择sigmoid")]),t._v(" "),s("li",[t._v("多分类问题选择softmax")]),t._v(" "),s("li",[t._v("回归问题选择identity激活函数（线性，未在此详述，CSDN查看）")])])])}),[],!1,null,null,null);s.default=e.exports}}]);