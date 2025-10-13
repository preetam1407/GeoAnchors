# GeoFormer — Preview (Q/Pred/Gold + Named Views)

## Example 0

**Q:** Question: Is <c1,CAM_BACK,1015.8,520.8> a traffic sign or a road barrier? Answer:
**Pred:** `No.`
**Gold:** `No.`

<table>
<tr>
<th style='text-align:center;padding:4px'>CAM_FRONT</th>
<th style='text-align:center;padding:4px'>CAM_FRONT_LEFT</th>
<th style='text-align:center;padding:4px'>CAM_FRONT_RIGHT</th>
</tr>
<tr>
<td style='text-align:center;padding:4px'><img src="../../data/nuscenes/samples/CAM_FRONT/n015-2018-08-03-15-00-36+0800__CAM_FRONT__1533279697262460.jpg" alt="" width="340"/></td>
<td style='text-align:center;padding:4px'><img src="../../data/nuscenes/samples/CAM_FRONT_LEFT/n015-2018-08-03-15-00-36+0800__CAM_FRONT_LEFT__1533279697254844.jpg" alt="" width="340"/></td>
<td style='text-align:center;padding:4px'><img src="../../data/nuscenes/samples/CAM_FRONT_RIGHT/n015-2018-08-03-15-00-36+0800__CAM_FRONT_RIGHT__1533279697270339.jpg" alt="" width="340"/></td>
</tr>
<tr>
<th style='text-align:center;padding:4px'>CAM_BACK</th>
<th style='text-align:center;padding:4px'>CAM_BACK_LEFT</th>
<th style='text-align:center;padding:4px'>CAM_BACK_RIGHT</th>
</tr>
<tr>
<td style='text-align:center;padding:4px'><img src="../../data/nuscenes/samples/CAM_BACK/n015-2018-08-03-15-00-36+0800__CAM_BACK__1533279697287525.jpg" alt="" width="340"/></td>
<td style='text-align:center;padding:4px'><img src="../../data/nuscenes/samples/CAM_BACK_LEFT/n015-2018-08-03-15-00-36+0800__CAM_BACK_LEFT__1533279697297423.jpg" alt="" width="340"/></td>
<td style='text-align:center;padding:4px'><img src="../../data/nuscenes/samples/CAM_BACK_RIGHT/n015-2018-08-03-15-00-36+0800__CAM_BACK_RIGHT__1533279697277893.jpg" alt="" width="340"/></td>
</tr>
</table>


## Example 1

**Q:** Question: What actions taken by the ego vehicle can lead to a collision with <c3,CAM_FRONT_RIGHT,1022.5,540.0>? Answer:
**Pred:** `Sharp right turn.`
**Gold:** `Moderate right turn.`

<table>
<tr>
<th style='text-align:center;padding:4px'>CAM_FRONT</th>
<th style='text-align:center;padding:4px'>CAM_FRONT_LEFT</th>
<th style='text-align:center;padding:4px'>CAM_FRONT_RIGHT</th>
</tr>
<tr>
<td style='text-align:center;padding:4px'><img src="../../data/nuscenes/samples/CAM_FRONT/n008-2018-07-26-12-13-50-0400__CAM_FRONT__1532621947162404.jpg" alt="" width="340"/></td>
<td style='text-align:center;padding:4px'><img src="../../data/nuscenes/samples/CAM_FRONT_LEFT/n008-2018-07-26-12-13-50-0400__CAM_FRONT_LEFT__1532621947154799.jpg" alt="" width="340"/></td>
<td style='text-align:center;padding:4px'><img src="../../data/nuscenes/samples/CAM_FRONT_RIGHT/n008-2018-07-26-12-13-50-0400__CAM_FRONT_RIGHT__1532621947170482.jpg" alt="" width="340"/></td>
</tr>
<tr>
<th style='text-align:center;padding:4px'>CAM_BACK</th>
<th style='text-align:center;padding:4px'>CAM_BACK_LEFT</th>
<th style='text-align:center;padding:4px'>CAM_BACK_RIGHT</th>
</tr>
<tr>
<td style='text-align:center;padding:4px'><img src="../../data/nuscenes/samples/CAM_BACK/n008-2018-07-26-12-13-50-0400__CAM_BACK__1532621947187558.jpg" alt="" width="340"/></td>
<td style='text-align:center;padding:4px'><img src="../../data/nuscenes/samples/CAM_BACK_LEFT/n008-2018-07-26-12-13-50-0400__CAM_BACK_LEFT__1532621947197405.jpg" alt="" width="340"/></td>
<td style='text-align:center;padding:4px'><img src="../../data/nuscenes/samples/CAM_BACK_RIGHT/n008-2018-07-26-12-13-50-0400__CAM_BACK_RIGHT__1532621947178113.jpg" alt="" width="340"/></td>
</tr>
</table>


## Example 2

**Q:** Question: Are there moving pedestrians to the back left of the ego car? Answer:
**Pred:** `Yes.`
**Gold:** `Yes.`

<table>
<tr>
<th style='text-align:center;padding:4px'>CAM_FRONT</th>
<th style='text-align:center;padding:4px'>CAM_FRONT_LEFT</th>
<th style='text-align:center;padding:4px'>CAM_FRONT_RIGHT</th>
</tr>
<tr>
<td style='text-align:center;padding:4px'><img src="../../data/nuscenes/samples/CAM_FRONT/n008-2018-08-27-11-48-51-0400__CAM_FRONT__1535385045512404.jpg" alt="" width="340"/></td>
<td style='text-align:center;padding:4px'><img src="../../data/nuscenes/samples/CAM_FRONT_LEFT/n008-2018-08-27-11-48-51-0400__CAM_FRONT_LEFT__1535385045504799.jpg" alt="" width="340"/></td>
<td style='text-align:center;padding:4px'><img src="../../data/nuscenes/samples/CAM_FRONT_RIGHT/n008-2018-08-27-11-48-51-0400__CAM_FRONT_RIGHT__1535385045520491.jpg" alt="" width="340"/></td>
</tr>
<tr>
<th style='text-align:center;padding:4px'>CAM_BACK</th>
<th style='text-align:center;padding:4px'>CAM_BACK_LEFT</th>
<th style='text-align:center;padding:4px'>CAM_BACK_RIGHT</th>
</tr>
<tr>
<td style='text-align:center;padding:4px'><img src="../../data/nuscenes/samples/CAM_BACK/n008-2018-08-27-11-48-51-0400__CAM_BACK__1535385045537558.jpg" alt="" width="340"/></td>
<td style='text-align:center;padding:4px'><img src="../../data/nuscenes/samples/CAM_BACK_LEFT/n008-2018-08-27-11-48-51-0400__CAM_BACK_LEFT__1535385045547405.jpg" alt="" width="340"/></td>
<td style='text-align:center;padding:4px'><img src="../../data/nuscenes/samples/CAM_BACK_RIGHT/n008-2018-08-27-11-48-51-0400__CAM_BACK_RIGHT__1535385045528113.jpg" alt="" width="340"/></td>
</tr>
</table>


## Example 3

**Q:** Question: What actions could the ego vehicle take based on <c1,CAM_FRONT_LEFT,515.8,584.2>? Why take this action and what's the probability? Answer:
**Pred:** `The action is to keep going at the`
**Gold:** `The action is to keep going at the same speed. The reason is that there is no safety issue. The probability of taking this action is high.`

<table>
<tr>
<th style='text-align:center;padding:4px'>CAM_FRONT</th>
<th style='text-align:center;padding:4px'>CAM_FRONT_LEFT</th>
<th style='text-align:center;padding:4px'>CAM_FRONT_RIGHT</th>
</tr>
<tr>
<td style='text-align:center;padding:4px'><img src="../../data/nuscenes/samples/CAM_FRONT/n008-2018-09-18-12-07-26-0400__CAM_FRONT__1537287212012404.jpg" alt="" width="340"/></td>
<td style='text-align:center;padding:4px'><img src="../../data/nuscenes/samples/CAM_FRONT_LEFT/n008-2018-09-18-12-07-26-0400__CAM_FRONT_LEFT__1537287212004799.jpg" alt="" width="340"/></td>
<td style='text-align:center;padding:4px'><img src="../../data/nuscenes/samples/CAM_FRONT_RIGHT/n008-2018-09-18-12-07-26-0400__CAM_FRONT_RIGHT__1537287212020482.jpg" alt="" width="340"/></td>
</tr>
<tr>
<th style='text-align:center;padding:4px'>CAM_BACK</th>
<th style='text-align:center;padding:4px'>CAM_BACK_LEFT</th>
<th style='text-align:center;padding:4px'>CAM_BACK_RIGHT</th>
</tr>
<tr>
<td style='text-align:center;padding:4px'><img src="../../data/nuscenes/samples/CAM_BACK/n008-2018-09-18-12-07-26-0400__CAM_BACK__1537287212037558.jpg" alt="" width="340"/></td>
<td style='text-align:center;padding:4px'><img src="../../data/nuscenes/samples/CAM_BACK_LEFT/n008-2018-09-18-12-07-26-0400__CAM_BACK_LEFT__1537287212047405.jpg" alt="" width="340"/></td>
<td style='text-align:center;padding:4px'><img src="../../data/nuscenes/samples/CAM_BACK_RIGHT/n008-2018-09-18-12-07-26-0400__CAM_BACK_RIGHT__1537287212028113.jpg" alt="" width="340"/></td>
</tr>
</table>


## Example 4

**Q:** Question: Would <c2,CAM_FRONT_LEFT,708.9,562.3> be in the moving direction of the ego vehicle? Answer:
**Pred:** `No.`
**Gold:** `No.`

<table>
<tr>
<th style='text-align:center;padding:4px'>CAM_FRONT</th>
<th style='text-align:center;padding:4px'>CAM_FRONT_LEFT</th>
<th style='text-align:center;padding:4px'>CAM_FRONT_RIGHT</th>
</tr>
<tr>
<td style='text-align:center;padding:4px'><img src="../../data/nuscenes/samples/CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402866662460.jpg" alt="" width="340"/></td>
<td style='text-align:center;padding:4px'><img src="../../data/nuscenes/samples/CAM_FRONT_LEFT/n015-2018-07-24-11-22-45+0800__CAM_FRONT_LEFT__1532402866654844.jpg" alt="" width="340"/></td>
<td style='text-align:center;padding:4px'><img src="../../data/nuscenes/samples/CAM_FRONT_RIGHT/n015-2018-07-24-11-22-45+0800__CAM_FRONT_RIGHT__1532402866670339.jpg" alt="" width="340"/></td>
</tr>
<tr>
<th style='text-align:center;padding:4px'>CAM_BACK</th>
<th style='text-align:center;padding:4px'>CAM_BACK_LEFT</th>
<th style='text-align:center;padding:4px'>CAM_BACK_RIGHT</th>
</tr>
<tr>
<td style='text-align:center;padding:4px'><img src="../../data/nuscenes/samples/CAM_BACK/n015-2018-07-24-11-22-45+0800__CAM_BACK__1532402866687525.jpg" alt="" width="340"/></td>
<td style='text-align:center;padding:4px'><img src="../../data/nuscenes/samples/CAM_BACK_LEFT/n015-2018-07-24-11-22-45+0800__CAM_BACK_LEFT__1532402866697423.jpg" alt="" width="340"/></td>
<td style='text-align:center;padding:4px'><img src="../../data/nuscenes/samples/CAM_BACK_RIGHT/n015-2018-07-24-11-22-45+0800__CAM_BACK_RIGHT__1532402866677893.jpg" alt="" width="340"/></td>
</tr>
</table>
