# ORB-SLAM2-SP
--- 
**Note !!  This project is a reverse example.**
- In offline operation, this code has higher robustness and performance.
- In real time operation, this code does not maintain its performance.

Therefore, this project intends to express the deficiencies of deep learning features in real-time applications such as SLAM.

### Experimental data:

ORB-SLAM2

<table border="1" bordercolor="black" width="600" cellspacing="0" cellpadding="5">
  <tr>
    <td rowspan="2"> Track coverage </td>
    <td rowspan="2"> offline </td>
    <td colspan="6" > real time </td>
  </tr>
  <tr>
    <td>100hz</td>
    <td>50hz</td>
    <td>10hz</td>
    <td>5hz</td>
    <td>2hz</td>
   </tr>
   <tr>
    <td> ORB-SLAM2 </td>
    <td>1</td>
    <td>0.22</td>
    <td>1</td>
    <td>1</td>
    <td>1</td>
    <td>1</td>
   </tr>
   <tr>
    <td> ORB-SLAM2-SP </td>
    <td>1</td>
    <td>Not initialized</td>
    <td>Not initialized</td>
    <td>0.18</td>
    <td>0.58</td>
    <td>1</td>
   </tr>
</table>


All data are averaged five times.

--- 
