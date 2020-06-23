#include <mitsuba/core/profiler.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/render/sampler.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _sampler-stratified:

Stratified sampler (:monosp:`stratified`)
-------------------------------------------

.. pluginparameters::

 * - sample_count
   - |int|
   - Number of samples per pixel. This number should be the square of a power of two (e.g. 4,
     16, 64, 256, 1024) (Default: 4)
 * - seed
   - |int|
   - Seed offset (Default: 0)
 * - jitter
   - |bool|
   - Adds additional random jitter withing the stratum (Default: True)

The stratified sample generator divides the domain into a discrete number of strata and produces
a sample within each one of them. This generally leads to less sample clumping when compared to
the independent sampler, as well as better convergence.

.. subfigstart::
.. subfigure:: ../../resources/data/docs/images/render/sampler_independent_16spp.jpg
   :caption: Independent sampler - 16 samples per pixel
.. subfigure:: ../../resources/data/docs/images/render/sampler_stratified_16spp.jpg
   :caption: Stratified sampler - 16 samples per pixel
.. subfigend::
   :label: fig-stratified-renders

.. subfigstart::
.. subfigure:: ../../resources/data/docs/images/sampler/stratified_1024_samples.svg
   :caption: 1024 samples projected onto the first two dimensions which are well distributed
             if we compare to the :monosp:`independent` sampler.
.. subfigure:: ../../resources/data/docs/images/sampler/stratified_64_samples_and_proj.svg
   :caption: 64 samples projected in 2D and on both 1D axis (top and right plot). Every strata
             contains a single sample creating a good distribution when projected in 2D. Projections
             on both 1D axis still exhibit sample clumping which will result in higher variance, for
             instance when sampling a thin streched rectangular area light.
.. subfigend::
   :label: fig-stratified-pattern

 */

template <typename Float, typename Spectrum>
class StratifiedSampler final : public RandomSampler<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(RandomSampler, m_sample_count, m_base_seed, m_rng,
                    check_rng, m_samples_per_wavefront, m_wavefront_count, wavefront_size)
    MTS_IMPORT_TYPES()

    StratifiedSampler(const Properties &props = Properties()) : Base(props) {
        m_jitter = props.bool_("jitter", true);

        // Make sure sample_count is power of two and square (e.g. 4, 16, 64, 256, 1024, ...)
        m_resolution = 2;
        while (sqr(m_resolution) < m_sample_count)
            m_resolution = math::round_to_power_of_two(++m_resolution);

        if (m_sample_count != sqr(m_resolution))
            Log(Warn, "Sample count should be square and power of two, rounding to %i", sqr(m_resolution));

        m_sample_count = sqr(m_resolution);
        m_inv_sample_count = rcp(ScalarFloat(m_sample_count));
        m_inv_resolution   = rcp(ScalarFloat(m_resolution));

        // Default
        m_samples_per_wavefront = 1;
        m_wavefront_count = m_sample_count;

        m_dimension_index = 0;
        m_wavefront_index = -1;
    }

    ref<Sampler<Float, Spectrum>> clone() override {
        StratifiedSampler *sampler = new StratifiedSampler();
        sampler->m_jitter                = m_jitter;
        sampler->m_sample_count          = m_sample_count;
        sampler->m_inv_sample_count      = m_inv_sample_count;
        sampler->m_resolution            = m_resolution;
        sampler->m_inv_resolution        = m_inv_resolution;
        sampler->m_samples_per_wavefront = m_samples_per_wavefront;
        sampler->m_wavefront_count       = m_wavefront_count;
        sampler->m_base_seed             = m_base_seed;
        sampler->m_dimension_index       = 0u;
        sampler->m_wavefront_index       = -1;
        return sampler;
    }

    void seed(UInt64 seed_value) override {
        ScopedPhase scope_phase(ProfilerPhase::SamplerSeed);
        Base::seed(seed_value);

        m_dimension_index = 0u;
        m_wavefront_index = -1;

        if constexpr (is_dynamic_array_v<Float>) {
            UInt32 indices = arange<UInt32>(seed_value.size());

            // Get the seed value of the first sample for every pixel
            UInt32 sequence_idx = m_samples_per_wavefront * (indices / m_samples_per_wavefront);
            UInt32 sequence_seeds = gather<UInt32>(seed_value, sequence_idx);
            m_permutations_seed = sample_tea_32<UInt32>(UInt32(m_base_seed), sequence_seeds);

            m_wavefront_sample_offsets = indices % UInt32(m_samples_per_wavefront);
        } else {
            m_permutations_seed = sample_tea_32<UInt32>(m_base_seed, seed_value);
            m_wavefront_sample_offsets = 0;
        }
    }

    void prepare_wavefront() override {
        m_dimension_index = 0u;
        m_wavefront_index++;
        Assert(m_wavefront_index < m_wavefront_count);
    }

    Float next_1d(Mask active = true) override {
        Assert(m_wavefront_index > -1);
        check_rng(active);

        UInt32 sample_indices = m_wavefront_index + m_wavefront_sample_offsets;
        UInt32 perm_seed = m_permutations_seed + m_dimension_index++;

        Float p = permute(sample_indices, m_sample_count, perm_seed);

        Float j = m_jitter ? next_float<Float>(m_rng.get(), active) : 0.5f;

        return (p + j) * m_inv_sample_count;
    }

    Point2f next_2d(Mask active = true) override {
        Assert(m_wavefront_index > -1);
        check_rng(active);

        UInt32 sample_indices = m_wavefront_index + m_wavefront_sample_offsets;
        UInt32 perm_seed = m_permutations_seed + m_dimension_index++;

        UInt32 p = permute(sample_indices, m_sample_count, perm_seed);

        UInt32 x = p % UInt32(m_resolution),
               y = p / m_resolution;

        Float jx = 0.5f, jy = 0.5f;
        if (m_jitter) {
            jx = next_float<Float>(m_rng.get(), active);
            jy = next_float<Float>(m_rng.get(), active);
        }

        return Point2f(x + jx, y + jy) * m_inv_resolution;
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "StratifiedSampler[" << std::endl
            << "  sample_count = " << m_sample_count << std::endl
            << "  jitter = " << m_jitter << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    bool m_jitter;

    /// Stratification grid resolution
    ScalarUInt32 m_resolution;
    ScalarFloat m_inv_resolution;
    ScalarFloat m_inv_sample_count;

    /// Sampler state
    ScalarUInt32 m_dimension_index;
    ScalarInt32  m_wavefront_index;

    UInt32 m_permutations_seed;
    UInt32 m_wavefront_sample_offsets;
};

MTS_IMPLEMENT_CLASS_VARIANT(StratifiedSampler, Sampler)
MTS_EXPORT_PLUGIN(StratifiedSampler, "Stratified Sampler");
NAMESPACE_END(mitsuba)
