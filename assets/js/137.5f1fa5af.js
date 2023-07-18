(window.webpackJsonp=window.webpackJsonp||[]).push([[137],{466:function(t,a,r){"use strict";r.r(a);var s=r(4),v=Object(s.a)({},(function(){var t=this,a=t._self._c;return a("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[a("h2",{attrs:{id:"路由链接"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#路由链接"}},[t._v("#")]),t._v(" 路由链接")]),t._v(" "),a("p",[a("img",{attrs:{src:"/more/Pasted%20image%2020220530190556.png",alt:"avatar"}})]),t._v(" "),a("p",[t._v("在路由网络中，物理线路直连意味着直连的双方能感知到对方存在。\n但假如我希望上图PC0与PC1进行聊天，又该如何处理呢？最直接方法就是他俩直连一条线，但这压根不可能。")]),t._v(" "),a("p",[t._v("因此我们更倾向于使用"),a("strong",[t._v("静态")]),t._v("和"),a("strong",[t._v("动态")]),t._v("路由实现。")]),t._v(" "),a("h4",{attrs:{id:"静态路由"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#静态路由"}},[t._v("#")]),t._v(" 静态路由")]),t._v(" "),a("p",[t._v("正如上面所说，需要设备物理线路直连时才能感知对方存在，假如我们想让PC0与PC1对话，由于他俩并非直连，因此需要告知router如何传递信息")]),t._v(" "),a("p",[t._v("人为地手动设置就是静态路由，如果途中IP变更，意味着静态路由也必须重设。")]),t._v(" "),a("p",[t._v("我们需要在Router0中设置静态路由，告知Router0如何处理目的地为34.72.2.0/24网络的请求。我们需要在配置模式下使用"),a("code",[t._v("ip route [目的地网段] [子网掩码] [下一跳地址]")]),t._v("进行配置\n例如在Router0中设置"),a("code",[t._v("ip route 34.72.2.0 255.255.255.0 134.72.1.2")]),t._v("。\n这样，当PC0发送请求给34.72.2.0/24网段时，Router0就会将该请求转给Router1的134.72.1.2接口，由于Router1与PC1直连，因此Router1知道如何与PC1链接。")]),t._v(" "),a("p",[t._v("静态路由优点是节约设备开销，但需要人为地维护静态路由，因此只适合小型网络结构。")]),t._v(" "),a("h4",{attrs:{id:"动态路由"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#动态路由"}},[t._v("#")]),t._v(" 动态路由")]),t._v(" "),a("p",[t._v("通过协议赋予节点“对话”的能力，彼此间可知道对方所拥有的路由映射表，从而实现全网共通，但缺点是需要耗费设备性能")]),t._v(" "),a("h6",{attrs:{id:"动态路由协议"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#动态路由协议"}},[t._v("#")]),t._v(" 动态路由协议")]),t._v(" "),a("p",[a("img",{attrs:{src:"/more/Pasted%20image%2020220530193007.png",alt:"avatar"}})]),t._v(" "),a("h6",{attrs:{id:"距离矢量"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#距离矢量"}},[t._v("#")]),t._v(" 距离矢量")]),t._v(" "),a("p",[t._v("距离矢量衡量一条路由路线好坏标准是跳数，也就是根据中转节点的个数判断好坏。")]),t._v(" "),a("p",[t._v("距离矢量的特点是周期性更新（广播）整个路由表")]),t._v(" "),a("p",[a("img",{attrs:{src:"/more/Pasted%20image%2020220530193421.png",alt:"avatar"}})]),t._v(" "),a("p",[t._v("路由器在初始启动时，只能感知与自己直连的网络。\n当启动距离矢量协议后，路由间会以广播形式交换彼此的路由表")]),t._v(" "),a("p",[a("img",{attrs:{src:"/more/Pasted%20image%2020220530193554.png",alt:"avatar"}})]),t._v(" "),a("p",[a("img",{attrs:{src:"/more/Pasted%20image%2020220530193741.png",alt:"avatar"}})]),t._v(" "),a("h6",{attrs:{id:"度量值"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#度量值"}},[t._v("#")]),t._v(" 度量值")]),t._v(" "),a("p",[a("img",{attrs:{src:"/more/Pasted%20image%2020220530194018.png",alt:"avatar"}})]),t._v(" "),a("p",[t._v("RIP以跳数作为度量值(metric)，RIP会选择跳数最少的路线作为路由路线\n当且仅当最短路径down后，下方路线才作为备用路线访问")]),t._v(" "),a("p",[t._v("我们可以通过"),a("code",[t._v("show ip route")]),t._v("查看路由配置，其中R开头的就是RIP")]),t._v(" "),a("p",[t._v("那么问题来了，倘若路由器A运行RIP协议，C运行OSPF协议，两者都知道路由K的位置，此时返回给路由B（B会RIP和OSPF），那么B会采取谁的路由？\n为应对这种情况的发生，每个路由协议都有AD值，当返回的来源是两种不同协议的结果，那么就会根据AD值决定，AD值越小越会被选择。\n"),a("img",{attrs:{src:"/more/Pasted%20image%2020220530195452.png",alt:"avatar"}})]),t._v(" "),a("h6",{attrs:{id:"rip协议更新流程"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#rip协议更新流程"}},[t._v("#")]),t._v(" RIP协议更新流程")]),t._v(" "),a("p",[a("img",{attrs:{src:"/more/Pasted%20image%2020220530195746.png",alt:"avatar"}})]),t._v(" "),a("p",[t._v("所谓逐跳更新是指，外部传给RouterB的路由表，B会先学习，然后再将学习后的结果转发给其他路由，而并非收到后立刻转发。")]),t._v(" "),a("h6",{attrs:{id:"潜在问题-环路"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#潜在问题-环路"}},[t._v("#")]),t._v(" 潜在问题 - 环路")]),t._v(" "),a("p",[a("img",{attrs:{src:"/more/Pasted%20image%2020220530200014.png",alt:"avatar"}})]),t._v(" "),a("p",[t._v("当10.4.0.0断线后，由于RIP是广播形式传递各自路由表，B的路由表中包含了10.4.0.0的路线，但此时由于C与10.4.0.0是直连因此能立刻感知10.4.0.0断线了，但由于B的路由表没有得到及时更新，让C误以为10.4.0.0是从B的方向过来的新路由，然后将表错误地更新了。\n"),a("img",{attrs:{src:"/more/Pasted%20image%2020220530200442.png",alt:"avatar"}})]),t._v(" "),a("p",[t._v("妥协方法：设置最大跳数，超过的直接无视，但会导致部分大跳路由永远无法到达")]),t._v(" "),a("p",[a("strong",[t._v("水平分割")]),t._v("\n路由器只发送其他路由器没有的路由表，C是10.4.0.0的发送源，因此B不会再向C发回10.4.0.0的路由表\n"),a("strong",[t._v("路由中毒")]),t._v("\n当路由器感知到拓扑变更时（不可用），会发送不可达信息给其他路由以便及时更新。\n"),a("strong",[t._v("抑制计时器")]),t._v("\n当路由短时间内接收到大量更差的路由表时，会暂时运行旧的路由表，等待更差路由的拥有者发送澄清信息，如果在规定时间内没收到澄清信息，则采用更差的路由表作为新路由表。")]),t._v(" "),a("h6",{attrs:{id:"配置rip"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#配置rip"}},[t._v("#")]),t._v(" 配置RIP")]),t._v(" "),a("p",[t._v("在config模式下\n启动RIP协议 - "),a("code",[t._v("router rip")]),t._v("\n宣告指定的直连网络接口 - "),a("code",[t._v("network [直连网段]")]),t._v("\n![[Pasted image 20220530203513.png]]\nRouter-0:")]),t._v(" "),a("ul",[a("li",[t._v("router rip")]),t._v(" "),a("li",[t._v("network 134.172.1.0")]),t._v(" "),a("li",[t._v("network 10.0.0.0\nRouter-1:")]),t._v(" "),a("li",[t._v("router rip")]),t._v(" "),a("li",[t._v("network 134.172.1.0")]),t._v(" "),a("li",[t._v("network 34.72.2.0")])]),t._v(" "),a("h4",{attrs:{id:"常用指令"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#常用指令"}},[t._v("#")]),t._v(" 常用指令")]),t._v(" "),a("div",{staticClass:"language- line-numbers-mode"},[a("pre",{pre:!0,attrs:{class:"language-text"}},[a("code",[t._v("show ? - 查看show相关指令的参数\nshow ip int b - 查看接口的ip\nshow ip route - 查看路由配置\n")])]),t._v(" "),a("div",{staticClass:"line-numbers-wrapper"},[a("span",{staticClass:"line-number"},[t._v("1")]),a("br"),a("span",{staticClass:"line-number"},[t._v("2")]),a("br"),a("span",{staticClass:"line-number"},[t._v("3")]),a("br")])])])}),[],!1,null,null,null);a.default=v.exports}}]);