// Loaded from main by URL: the same two files the diffusion README uses.
export const DiffusionLogo = ({ width = 520 }) => {
  const base = "https://raw.githubusercontent.com/sgl-project/sglang/main/assets/sglang-diffusion/";
  const style = { width: "100%", maxWidth: width, margin: "0 auto 1.5rem" };

  return (
    <div className="not-prose">
      <img className="block dark:hidden" src={base + "logo.svg"} alt="SGLang Diffusion" style={style} />
      <img className="hidden dark:block" src={base + "logo-dark.svg"} alt="SGLang Diffusion" style={style} />
    </div>
  );
};
