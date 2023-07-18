(window.webpackJsonp=window.webpackJsonp||[]).push([[120],{450:function(a,v,t){"use strict";t.r(v);var r=t(4),_=Object(r.a)({},(function(){var a=this,v=a._self._c;return v("ContentSlotsDistributor",{attrs:{"slot-key":a.$parent.slotKey}},[v("h4",{attrs:{id:"简介"}},[v("a",{staticClass:"header-anchor",attrs:{href:"#简介"}},[a._v("#")]),a._v(" 简介")]),a._v(" "),v("p",[a._v("在先前介绍中，我们了解到，同一广播域下的设备在一开始并不知道对方的存在，因此需要通过交换机进行泛洪后才能探知到对方的存在，但一旦网络规模变得庞大（成百上千的PC）时，无止境的泛洪会严重影响线路效率。因此我们需要一个手段来限制广播域的规模，这种技术称为VLAN。")]),a._v(" "),v("p",[v("img",{attrs:{src:"/more/Pasted%20image%2022021216180353.png",alt:"avatar"}})]),a._v(" "),v("p",[a._v("如上图所示，数据从主机A发出时并不会携带VLAN标签，当数据到达PVID2接口时，会被自动标记为VLAN2数据，然后经由Trunk链路传递给SWB，然后再PVID2处，VLAN2标签被剥离，然后再传递给主机C")]),a._v(" "),v("h4",{attrs:{id:"实验配置"}},[v("a",{staticClass:"header-anchor",attrs:{href:"#实验配置"}},[a._v("#")]),a._v(" 实验配置")]),a._v(" "),v("p",[v("img",{attrs:{src:"/more/Pasted%20image%2020221216182624.png",alt:"avatar"}})]),a._v(" "),v("p",[a._v("假设我们有以上的拓扑环境，PC1和PC3同属VLAN10，PC2和PC4同属VLAN20.")]),a._v(" "),v("p",[a._v("首先我们需要在LSW1中划分VLAN，我们可以通过"),v("code",[a._v("[SW1] vlan batch 10 20")]),a._v("命令创建VLAN10和VLAN20。然后在各自对于的接口，如LSW1与PC1相连的接口设置链路模式。\n"),v("code",[a._v("[SW1-GigabitEthernet0/0/1]port link-type access")]),a._v("，这样我们就能把链路设置为ACCESS模式了，然后输入"),v("code",[a._v("[SW1-GigabitEthernet0/0/1]port default vlan 10")]),a._v("将该接口输出的数据设置为VLAN10，以此类推设置其余的几台PC与交换机之间的关系。")]),a._v(" "),v("p",[a._v("然后进入LSW1和LSW2相连的接口，分别将接口设置为Trunk模式：\n"),v("code",[a._v("port link-type trunk")]),a._v("和"),v("code",[a._v("port trunk allow-pass vlan 10")]),a._v("，这样就能告知交换机将该线路模式设置为trunk并且允许vlan10访问")]),a._v(" "),v("h4",{attrs:{id:"三个常见接口"}},[v("a",{staticClass:"header-anchor",attrs:{href:"#三个常见接口"}},[a._v("#")]),a._v(" 三个常见接口")]),a._v(" "),v("p",[a._v("在交换机中，有三个常见的配置VLAN的接口，分别是Trunk、Hybrid和Access")]),a._v(" "),v("h6",{attrs:{id:"trunk"}},[v("a",{staticClass:"header-anchor",attrs:{href:"#trunk"}},[a._v("#")]),a._v(" Trunk")]),a._v(" "),v("p",[a._v("常见于交换机之间，Trunk允许多个VLAN通过，并且能够打上VLAN标签，")]),a._v(" "),v("h6",{attrs:{id:"hybrid"}},[v("a",{staticClass:"header-anchor",attrs:{href:"#hybrid"}},[a._v("#")]),a._v(" Hybrid")]),a._v(" "),v("p",[a._v("常见于交换机之间，Hybrid也允许多个VLAN通过，并且能够打上VLAN标签，并且能剥离多个标签（不同PVID的也能剥离）")]),a._v(" "),v("p",[v("img",{attrs:{src:"/more/Pasted%20image%2020221217150527.png",alt:"avatar"}})]),a._v(" "),v("p",[a._v("假设我们有以上的简单拓扑，我们可以通过给以上的交换机所对应的接口设置不同的VLAN隔绝广播域，在Hybrid中步骤如下:")]),a._v(" "),v("ol",[v("li",[v("code",[a._v("vlan batch 10 20 - 开辟两个vlan")])]),a._v(" "),v("li",[v("code",[a._v("interface g 0/0/1 - 进入g0/0/1接口")])]),a._v(" "),v("li",[v("code",[a._v("port link-type hybrid - 接口类型设置为hybrid模式")])]),a._v(" "),v("li",[v("code",[a._v("port hybrid pvid vlan 10 - 为接口设置vlan 10（只有打上10标签的数据包才能进入）")])]),a._v(" "),v("li",[v("code",[a._v("port hybrid untagged vlan 10 - 给发出的包打上vlan10标签")])])]),a._v(" "),v("p",[a._v("对g0/0/2接口设置相同的步骤即可，此时可以发现，两台主机已经无法完成通信，因为ARP报文在进入对方区域时并未携带对应的标签"),v("code",[a._v("vlan 10")]),a._v("或"),v("code",[a._v("vlan 20")]),a._v("，因此ARP报文最终无法抵达目标主机。")]),a._v(" "),v("p",[a._v("基于Hybrid特性，我们可以在接口处为数据包打上多个vlan的标签使其在网络间畅通无阻，如在"),v("code",[a._v("GE0/0/1")]),a._v("上可以使用"),v("code",[a._v("port hybrid untagged vlan 20")]),a._v("，这样发出的包会包含"),v("code",[a._v("vlan 10 20")]),a._v("两个标签，但值得注意的是，目标的接口也必须执行相似的操作，否则回传的包会因缺失标签无法返回。")]),a._v(" "),v("p",[a._v("在华为设备中，默认模式是hybrid，但该协议是华为的私有协议，这意味着在其他品牌的设备中并不存在。")]),a._v(" "),v("p",[a._v("此外，一旦接口设置了"),v("code",[a._v("port hybrid pvid vlan 10")]),a._v("就意味着有且仅有vlan 10标签的包能通过。")]),a._v(" "),v("h6",{attrs:{id:"access"}},[v("a",{staticClass:"header-anchor",attrs:{href:"#access"}},[a._v("#")]),a._v(" Access")]),a._v(" "),v("p",[a._v("常见于设备与交换机之间，Access能够剥离与自己同PVID的标签，并且发出时也会打上相应的PVID标签。")]),a._v(" "),v("h4",{attrs:{id:"三层交换设备"}},[v("a",{staticClass:"header-anchor",attrs:{href:"#三层交换设备"}},[a._v("#")]),a._v(" 三层交换设备")]),a._v(" "),v("p",[a._v("通常情况下，Switch还支持三层交换功能，试想以下情况\n"),v("img",{attrs:{src:"/more/Pasted%20image%2020221217175259.png",alt:"avatar"}})]),a._v(" "),v("p",[a._v("我们希望主机1和主机2之间能够交流，但运用之前所学知识是办不到的，因为主机1和主机2并不在同一个网段中，当主机1发送的包目的IP是其他网段，则会自动导向路由器，因为这很明显是属于外部网段的，所以就自动去找了路由器进行转发，但事实上这并非我们想要的结果。因此我们需要引入一个概念 —— VLANIF。")]),a._v(" "),v("p",[a._v("VLANIF是内置于交换机中的虚拟路由，他能做到路由器的绝大多数功能。\n"),v("img",{attrs:{src:"/more/Pasted%20image%2020221217175702.png",alt:"avatar"}})]),a._v(" "),v("p",[a._v("这张图详解了VLANIF在交换机中的存在，我们可以看到，同vlan的设备会被接入到相同的虚拟交换机中（如SW10），然后连接至一台虚拟路由器中，虚拟路由器和真实路由器的接口一样拥有IP地址，通过一台内置的虚拟路由器就能实现交换机下两台不同网段的设备互相交流。")]),a._v(" "),v("p",[a._v("还是以此图为例：\n"),v("img",{attrs:{src:"/more/Pasted%20image%2020221217180130.png",alt:"avatar"}})]),a._v(" "),v("p",[a._v("为主机设置完IP地址后，我们划分了两个vlan，然后在交换机输入"),v("code",[a._v("int Vlanif 10")]),a._v("便可访问各个Vlan的vlanif，然后通过"),v("code",[a._v("ip addr [IP] [submask]")]),a._v("设置网关IP即可，由于我们有两个VLAN，因此同理还需要用"),v("code",[a._v("int Vlanif 20")]),a._v("设置对应的网关IP。这样就能实现两个不同网段间通信了")])])}),[],!1,null,null,null);v.default=_.exports}}]);