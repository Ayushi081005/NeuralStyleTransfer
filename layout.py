"""
ui/layout.py
All Gradio UI components, CSS, and HTML strings.
"""

css = """
:root {
    --lilac-light:  #E8E0FF;
    --lilac-medium: #C5B3FF;
    --lilac-accent: #7C4DFF;
    --lilac-deep:   #5E35B1;
    --text-dark:    #483D8B;
}
.gradio-container {
    background: linear-gradient(135deg, var(--lilac-light) 0%, #F5F0FF 100%);
    color: var(--text-dark);
    max-width: 1200px;
    margin: 0 auto;
    font-family: 'Poppins', -apple-system, BlinkMacSystemFont, sans-serif;
}
#app-title {
    color: var(--lilac-deep); font-size: 44px; font-weight: 800;
    text-align: center; margin: 20px auto 5px;
}
#app-subtitle { text-align: center; margin-bottom: 25px; font-size: 17px; color: var(--text-dark); }
.app-section {
    border-radius: 20px;
    box-shadow: 0 8px 32px rgba(139,127,219,.15);
    transition: transform .3s, box-shadow .3s;
    margin-bottom: 1.5rem;
    background: white;
    overflow: hidden;
}
.app-section:hover { transform: translateY(-4px); box-shadow: 0 12px 48px rgba(139,127,219,.25); }
#about-section { background: linear-gradient(135deg,#FFF 0%,var(--lilac-light) 100%); padding: 2rem; text-align: center; }
#about-section h2 { color: var(--lilac-deep); font-size: 1.8rem; margin-bottom: 1rem; }
#about-section p  { color: var(--text-dark); font-size: 1.05rem; line-height: 1.6; max-width: 800px; margin: 1rem auto 0; }
.features-grid { display: grid; grid-template-columns: repeat(auto-fit,minmax(220px,1fr)); gap: 1.5rem; margin-top: 1.5rem; padding: 0 1rem; }
.feature-item  { background: var(--lilac-light); padding: 1.5rem; border-radius: 12px; text-align: center; transition: all .3s; }
.feature-item:hover { background: var(--lilac-medium); transform: translateY(-5px); }
.feature-icon  { font-size: 2.2rem; margin-bottom: .8rem; }
.feature-title { font-weight: 600; color: var(--lilac-deep); margin-bottom: .4rem; }
.feature-text  { color: var(--text-dark); font-size: .92rem; line-height: 1.5; }
#generate-btn  {
    background: linear-gradient(45deg,var(--lilac-accent),var(--lilac-deep)) !important;
    color: white !important; font-weight: 600 !important; font-size: 1.15rem !important;
    padding: .75rem 2rem !important; border-radius: 12px !important; border: none !important;
    transition: all .3s !important; box-shadow: 0 4px 15px rgba(124,77,255,.3) !important;
    margin: 1rem auto !important; display: block !important; width: fit-content !important;
}
#generate-btn:hover { transform: translateY(-3px) !important; box-shadow: 0 8px 25px rgba(124,77,255,.4) !important; }
.output-container { margin-top: 1.5rem; padding: 1.5rem; border-radius: 15px; background: var(--lilac-light); }
.pipeline-box {
    background: var(--lilac-light); border-radius: 12px; padding: 1.2rem 1.5rem;
    margin: 1rem 0; font-size: .95rem; line-height: 1.7; color: var(--text-dark); text-align: center;
}
.pipeline-box strong { color: var(--lilac-deep); }
.nst-options { background: var(--lilac-light); border-radius: 12px; padding: 1rem; margin-top: .5rem; }
#footer { text-align: center; padding: 1.5rem; color: var(--text-dark); font-size: .9rem; opacity: .7; margin-top: 1.5rem; }
@media(max-width:768px){ #app-title{font-size:32px;} .features-grid{grid-template-columns:1fr;} }
"""

header_html = (
    '<h1 id="app-title">Artistic Image Stylization</h1>'
    '<div id="app-subtitle">Using Digital Image Processing &amp; Neural Style Transfer</div>'
)

about_html = """
<div id="about-section" class="app-section">
  <h2>About This Project</h2>
  <p>
    This application demonstrates <strong>Artistic Image Stylization</strong> using both
    classical DIP techniques and Neural Style Transfer. The 6 preset styles use edge detection,
    spatial filtering, histogram equalisation, and colour transforms. The 7th option &mdash;
    <strong>Reference Style Transfer</strong> &mdash; uses a VGG19 neural network to transfer
    the artistic style of any reference image onto your photo.
  </p>
</div>
"""

features_html = """
<div class="app-section" style="padding:2rem;text-align:center;">
  <h2 style="color:var(--lilac-deep);font-size:1.8rem;margin-bottom:.5rem;">Techniques Implemented</h2>
  <div class="features-grid">
    <div class="feature-item">
      <div class="feature-icon">&#128269;</div>
      <div class="feature-title">Edge Detection</div>
      <div class="feature-text">Canny, Sobel, Laplacian + Otsu thresholding</div>
    </div>
    <div class="feature-item">
      <div class="feature-icon">&#127744;</div>
      <div class="feature-title">Spatial Filtering</div>
      <div class="feature-text">Bilateral, Mean-shift, Gaussian, Unsharp mask, Convolution kernels</div>
    </div>
    <div class="feature-item">
      <div class="feature-icon">&#127912;</div>
      <div class="feature-title">Histogram &amp; Colour</div>
      <div class="feature-text">CLAHE, K-means quantisation, HSV/LAB manipulation</div>
    </div>
    <div class="feature-item">
      <div class="feature-icon">&#129504;</div>
      <div class="feature-title">Neural Style Transfer</div>
      <div class="feature-text">VGG19 Gram matrix (Gatys et al.) + Adam optimisation + AMP/FP16</div>
    </div>
  </div>
</div>
"""

pipeline_html = """
<div class="pipeline-box">
  <strong>Preset Styles:</strong>
  Preprocessing &rarr; Edge Detection &rarr; Spatial Filtering &rarr; Colour Transform &rarr; Output<br>
  <strong>Reference Transfer:</strong>
  Content + Style &rarr; VGG19 Features &rarr; Gram Matrix Loss + Content Loss &rarr; Adam&nbsp;(+&nbsp;AMP) &rarr; Output
</div>
"""

footer_html = (
    '<div id="footer">'
    'Artistic Image Stylization | DIP + Neural Style Transfer | '
    'OpenCV, PyTorch &amp; Gradio'
    '</div>'
)
