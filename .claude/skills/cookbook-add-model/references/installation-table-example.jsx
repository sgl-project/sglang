<table style={{ width: "100%", borderCollapse: "collapse", tableLayout: "fixed" }}>
  <colgroup>
    <col style={{ width: "55%" }} />
    <col style={{ width: "45%" }} />
  </colgroup>
  <thead>
    <tr style={{ borderBottom: "2px solid #d55816" }}>
      <th style={{ textAlign: "left", padding: "10px 12px", fontWeight: 700, backgroundColor: "rgba(255,255,255,0.02)" }}>
        Hardware Platform
      </th>
      <th style={{ textAlign: "left", padding: "10px 12px", fontWeight: 700, backgroundColor: "rgba(255,255,255,0.05)" }}>
        Docker Image
      </th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style={{ padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)" }}>
        NVIDIA A100 / H100 / H200 / B200
      </td>
      <td style={{ padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)" }}>
        <code>lmsysorg/sglang:&lt;ver&gt;</code>
      </td>
    </tr>
    <tr>
      <td style={{ padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)" }}>
        NVIDIA B300 / GB300
      </td>
      <td style={{ padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)" }}>
        <code>lmsysorg/sglang:&lt;ver&gt;-cu130</code>
      </td>
    </tr>
    <tr>
      <td style={{ padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)" }}>
        AMD MI300X / MI325X
      </td>
      <td style={{ padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)" }}>
        <code>lmsysorg/sglang:&lt;ver&gt;-rocm720-mi30x</code>
      </td>
    </tr>
    <tr>
      <td style={{ padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)" }}>
        AMD MI350X / MI355X
      </td>
      <td style={{ padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)" }}>
        <code>lmsysorg/sglang:&lt;ver&gt;-rocm720-mi35x</code>
      </td>
    </tr>
  </tbody>
</table>
