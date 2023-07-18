(window.webpackJsonp=window.webpackJsonp||[]).push([[138],{467:function(t,a,r){"use strict";r.r(a);var v=r(4),s=Object(v.a)({},(function(){var t=this,a=t._self._c;return a("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[a("h2",{attrs:{id:"交换基础-vlan-trunk-vtp"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#交换基础-vlan-trunk-vtp"}},[t._v("#")]),t._v(" 交换基础 VLAN TRUNK VTP")]),t._v(" "),a("h4",{attrs:{id:"园区网结构"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#园区网结构"}},[t._v("#")]),t._v(" 园区网结构")]),t._v(" "),a("p",[a("img",{attrs:{src:"/more/Pasted%20image%2020220601154135.png",alt:"avatar"}})]),t._v(" "),a("h4",{attrs:{id:"交换机"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#交换机"}},[t._v("#")]),t._v(" 交换机")]),t._v(" "),a("p",[t._v("功能：")]),t._v(" "),a("ul",[a("li",[t._v("学习地址 : MAC地址，交换机学习完地址后会构成一张MAC表，交换机根据这张MAC地址表进行数据转发。注意，MAC表在断电后会被清空")]),t._v(" "),a("li",[t._v("转发与过滤 : 二层转发，当交换机收到一个帧时，会根据数据头的MAC地址从MAC地址表中查找对应的去处")]),t._v(" "),a("li",[t._v("环路的避免")])]),t._v(" "),a("h6",{attrs:{id:"工作原理"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#工作原理"}},[t._v("#")]),t._v(" 工作原理")]),t._v(" "),a("p",[t._v("在交换机开机时，内置MAC表为空，假设现在有ABCD四台设备接入交换机。在一开始，假设只有A被录入MAC地址表中。")]),t._v(" "),a("table",[a("thead",[a("tr",[a("th",[t._v("接口")]),t._v(" "),a("th",[t._v("MAC地址")])])]),t._v(" "),a("tbody",[a("tr",[a("td",[t._v("int0/0")]),t._v(" "),a("td",[t._v("MAC-A")])]),t._v(" "),a("tr",[a("td",[t._v("#")]),t._v(" "),a("td",[t._v("#")])])])]),t._v(" "),a("p",[t._v("此时，A打算发消息给D，由于MAC地址表中没有D的信息，因此交换机会采取"),a("strong",[t._v("泛洪")]),t._v("策略，即广播给所有接入设备。BC在拆包后发现不是给自己的，就会丢弃，而D会继续拆包。由于通信是双向的，因此D需要回复A告知其接收到了包，在D发送给A的过程中，经过交换机时，D的MAC信息也会被录入。此时表会变为")]),t._v(" "),a("table",[a("thead",[a("tr",[a("th",[t._v("接口")]),t._v(" "),a("th",[t._v("MAC地址")])])]),t._v(" "),a("tbody",[a("tr",[a("td",[t._v("int0/0")]),t._v(" "),a("td",[t._v("MAC-A")])]),t._v(" "),a("tr",[a("td",[t._v("int0/1")]),t._v(" "),a("td",[t._v("MAC-D")])])])]),t._v(" "),a("h4",{attrs:{id:"mac地址"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#mac地址"}},[t._v("#")]),t._v(" MAC地址")]),t._v(" "),a("p",[a("img",{attrs:{src:"/more/Pasted%20image%2020220601160124.png",alt:"avatar"}})]),t._v(" "),a("h4",{attrs:{id:"vlan"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#vlan"}},[t._v("#")]),t._v(" VLAN")]),t._v(" "),a("p",[a("img",{attrs:{src:"/more/Pasted%20image%2020220601162419.png",alt:"avatar"}}),t._v(" "),a("img",{attrs:{src:"/more/Pasted%20image%2020220601195916.png",alt:"avatar"}})]),t._v(" "),a("p",[t._v("Static: 手动分配交换机固定端口作为VLAN\nDynamic: 根据接入设备的MAC地址，动态将端口分配给VLAN")]),t._v(" "),a("p",[a("img",{attrs:{src:"/more/Pasted%20image%2020220601202823.png",alt:"avatar"}})]),t._v(" "),a("h6",{attrs:{id:"特点"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#特点"}},[t._v("#")]),t._v(" 特点")]),t._v(" "),a("p",[t._v("一个VLAN中所有设备都在同一广播域内，广播不能跨VLAN传播\n一个VLAN为一个逻辑子网，由被配置为此VLAN成员的设备组成，不同VLAN间需要通过路由器实现通信\nVLAN中成员多基于Switch端口号码，划分VLAN就是对Switch接口的划分\nVLAN工作于OSI参考模型第二层")]),t._v(" "),a("h4",{attrs:{id:"trunk"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#trunk"}},[t._v("#")]),t._v(" Trunk")]),t._v(" "),a("p",[t._v("一条链路，需要承载多个VLAN信息时，需要用Trunk实现，一般见于交换机之间或交换机和路由器之间")]),t._v(" "),a("h6",{attrs:{id:"isl"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#isl"}},[t._v("#")]),t._v(" ISL")]),t._v(" "),a("p",[t._v("ISL以封装的方式，在原始数据帧前封装一个ISL数据头。该协议是属于思科的协议，因此使用有局限性。\n"),a("img",{attrs:{src:"/more/Pasted%20image%2020220601202154.png",alt:"avatar"}})]),t._v(" "),a("h6",{attrs:{id:"_802-1q"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#_802-1q"}},[t._v("#")]),t._v(" 802.1Q")]),t._v(" "),a("p",[t._v("802.1Q是公有协议，在原始以太网数据帧中插入字段并将原始FCS（帧校验值）移除，然后重新计算新的校验值。")]),t._v(" "),a("h4",{attrs:{id:"vtp"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#vtp"}},[t._v("#")]),t._v(" VTP")]),t._v(" "),a("p",[t._v("VLAN Trunking Protocol是一个能够宣发VLAN配置信息的信息协议，通过一个共有的管理域，维持VLAN配置信息的一致性。VTP只能在主干端口发送要宣告的信息，支持混合的介质主干连接")]),t._v(" "),a("h6",{attrs:{id:"模式"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#模式"}},[t._v("#")]),t._v(" 模式")]),t._v(" "),a("p",[a("img",{attrs:{src:"/more/Pasted%20image%2020220601203530.png",alt:"avatar"}})])])}),[],!1,null,null,null);a.default=s.exports}}]);