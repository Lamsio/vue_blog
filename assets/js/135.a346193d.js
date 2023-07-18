(window.webpackJsonp=window.webpackJsonp||[]).push([[135],{464:function(a,t,r){"use strict";r.r(t);var s=r(4),e=Object(s.a)({},(function(){var a=this,t=a._self._c;return t("ContentSlotsDistributor",{attrs:{"slot-key":a.$parent.slotKey}},[t("h2",{attrs:{id:"ospf协议"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#ospf协议"}},[a._v("#")]),a._v(" OSPF协议")]),a._v(" "),t("p",[a._v("OSPF协议是一款开放式协议，相比于EIGRP仅限思科使用，OSPF协议允许其他厂商使用")]),a._v(" "),t("p",[a._v("OSPF采用SPF算法计算到达目的地的最短路径")]),a._v(" "),t("ul",[t("li",[a._v("什么叫链路（LINK）？=> 路由器接口")]),a._v(" "),t("li",[a._v("什么叫状态（STATE）？=> 描述接口以及其他邻居路由之间的关系")])]),a._v(" "),t("h4",{attrs:{id:"ospf-metric"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#ospf-metric"}},[a._v("#")]),a._v(" OSPF metric")]),a._v(" "),t("p",[a._v("每个路由都把自己当作根路由，并给予累计成本（cost值）来计算到达目的地的最短路径")]),a._v(" "),t("p",[a._v("COST = 参考宽带(10^8) / 接口带宽(b/s)")]),a._v(" "),t("p",[t("img",{attrs:{src:"/more/Pasted%20image%2020220601132152.png",alt:"avatar"}})]),a._v(" "),t("h4",{attrs:{id:"ospf报文"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#ospf报文"}},[a._v("#")]),a._v(" OSPF报文")]),a._v(" "),t("p",[t("img",{attrs:{src:"/more/Pasted%20image%2020220601132906.png",alt:"avatar"}}),a._v("\nLSDB: LSDB 是对整个网络拓扑结构和网段信息的汇总，同步完 LSDB 后，所有路由器对网络的认识是一样的\nLSA: 使用 "),t("strong",[a._v("LSA")]),a._v("（ Link State Advertisement ，链路状态通告）来装载和传输链路状态信息。LSA 需要描述邻接路由器信息、直连链路信息、跨区域信息等，所以定义了多种类型的 LSA 。\n"),t("img",{attrs:{src:"/more/Pasted%20image%2020221215203433.png",alt:"avatar"}}),a._v("\n路由器间发送DD报文时，需要先交换各自的Router ID用于决定主从关系，主从关系用于决定发送的先后次序。")]),a._v(" "),t("h4",{attrs:{id:"ospf区域"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#ospf区域"}},[a._v("#")]),a._v(" OSPF区域")]),a._v(" "),t("p",[t("img",{attrs:{src:"/more/Pasted%20image%2020220601133936.png",alt:"avatar"}}),a._v("\n由于每台路由器都包含了整个区域的详细信息，随着网络规模变大，会给路由器带来不少的负荷，因此我们需要进行划区操作，而有一台特殊路由器用于连接两个不同区域的网络")]),a._v(" "),t("p",[t("img",{attrs:{src:"/more/Pasted%20image%2020221215211549.png",alt:"avatar"}}),a._v("\n以上图为例，RTA、RTB、RTC作为ABR连接着三个各自的区域，ABR路由器需要维护两个LSDB，此外ABR路由器必须属于骨干网络，因此，当RTD尝试访问RTE时，就会向RTA发送数据由RTA将数据交给RTC然后再转交给RTE")]),a._v(" "),t("h4",{attrs:{id:"ospf的三张表"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#ospf的三张表"}},[a._v("#")]),a._v(" OSPF的三张表")]),a._v(" "),t("p",[t("img",{attrs:{src:"/more/Pasted%20image%2020220601133535.png",alt:"avatar"}})]),a._v(" "),t("h4",{attrs:{id:"ospf的基本运行步骤"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#ospf的基本运行步骤"}},[a._v("#")]),a._v(" OSPF的基本运行步骤")]),a._v(" "),t("p",[t("img",{attrs:{src:"/more/Pasted%20image%2020220601135204.png",alt:"avatar"}}),a._v(" "),t("img",{attrs:{src:"/more/Pasted%20image%2020220601135301.png",alt:"avatar"}})]),a._v(" "),t("h4",{attrs:{id:"ospf网络类型"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#ospf网络类型"}},[a._v("#")]),a._v(" OSPF网络类型")]),a._v(" "),t("p",[t("img",{attrs:{src:"/more/Pasted%20image%2020220601135501.png",alt:"avatar"}}),a._v("\n根据数据链路层的封装，来判断处于OSPF哪一种网络类型当中")]),a._v(" "),t("h4",{attrs:{id:"lsa泛洪"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#lsa泛洪"}},[a._v("#")]),a._v(" LSA泛洪")]),a._v(" "),t("p",[a._v("在与其他路由器通信前，需要建立路由关系。在BMA网络类型中，该损耗尤为明显，一台路由器需要与n-1台路由器进行通信、交换路由信息。\n当某台路由器中断时，也必须向其他所有路由器发送信息告知，因此在MA网络类型中，我们提出了新的概念 —— DR 和 BDR")]),a._v(" "),t("p",[t("img",{attrs:{src:"/more/Pasted%20image%2020220601143939.png",alt:"avatar"}})]),a._v(" "),t("h6",{attrs:{id:"dr"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#dr"}},[a._v("#")]),a._v(" DR")]),a._v(" "),t("p",[a._v("指定路由器，即选择某台路由作为代表，其他路由仅与其交流。当下属某条线路down了，该线路只需要与代理路由器进行交流即可，并由代理路由器告知其他下属路由器")]),a._v(" "),t("h6",{attrs:{id:"bdr"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#bdr"}},[a._v("#")]),a._v(" BDR")]),a._v(" "),t("p",[a._v("备份指定路由器，在DR基础上再搭建一个BDR作为DR的备用，避免DR挂了网络瘫痪")]),a._v(" "),t("h6",{attrs:{id:"ma拓扑变更"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#ma拓扑变更"}},[a._v("#")]),a._v(" MA拓扑变更")]),a._v(" "),t("p",[a._v("在MA环境下，当下属路由的拓扑发生变更时，下属路由会向224.0.0.6发送通知，DR和BDR监听该地址")]),a._v(" "),t("p",[a._v("DR BDR在收到变更通知后，会利用组播地址224.0.0.5通知其他下属路由，所有下属路由监听224.0.0.5这一组播地址")]),a._v(" "),t("p",[t("img",{attrs:{src:"/more/Pasted%20image%2020220601145706.png",alt:"avatar"}})]),a._v(" "),t("h4",{attrs:{id:"ospf配置"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#ospf配置"}},[a._v("#")]),a._v(" OSPF配置")]),a._v(" "),t("p",[t("code",[a._v("Router(config)#router ospf [process-id]")]),a._v(" 开启OSPF进程\n"),t("code",[a._v("Router(config)#network address wildcard-mask area area-id")]),a._v(" 宣告特定的网络到OSPF区域\n"),t("img",{attrs:{src:"/more/Pasted%20image%2020220601153829.png",alt:"avatar"}}),a._v(" "),t("img",{attrs:{src:"/more/Pasted%20image%2020220601153807.png",alt:"avatar"}})])])}),[],!1,null,null,null);t.default=e.exports}}]);