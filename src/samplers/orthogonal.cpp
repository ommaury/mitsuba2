#include <mitsuba/core/profiler.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/render/sampler.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _sampler-orthogonal:

Orthogonal Array sampler (:monosp:`orthogonal`)
-----------------------------------------------

.. pluginparameters::

 * - sample_count
   - |int|
   - Number of samples per pixel. This value has to be the square of a prime number. (Default: 4)
 * - strength
   - |int|
   - Orthogonal array's strength (Default: 2)
 * - seed
   - |int|
   - Seed offset (Default: 0)
 * - jitter
   - |bool|
   - Adds additional random jitter withing the substratum (Default: True)

This plugin implements the Orthogonal Array sampler generator introduced by
:cite:`jarosz19orthogonal`. It generalizes correlated multi-jittered sampling to higher dimensions
by using *orthogonal arrays (OAs)* of strength :math:`s`. The strength property of OAs tells that
projecting the generated samples to any combination of :math:`s` dimensions will always result in
a well stratified pattern. In other words, when :math:`s=2` (default value), the high-dimentional
samples are simultaneously stratified in all 2D projections as if they had been produced by
correlated multi-jittered sampling. By construction, samples produced by this generator are also
well stratified when projected on both 1D axis.

.. subfigstart::
.. subfigure:: ../../resources/data/docs/images/render/sampler_independent_25spp.jpg
   :caption: Independent sampler - 25 samples per pixel
.. subfigure:: ../../resources/data/docs/images/render/sampler_orthogonal_25spp.jpg
   :caption: Orthogonal Array sampler - 25 samples per pixel
.. subfigend::
   :label: fig-orthogonal-renders

.. subfigstart::
.. subfigure:: ../../resources/data/docs/images/sampler/orthogonal_1369_samples.svg
   :caption: 1369 samples projected onto the first two dimensions.
.. subfigure:: ../../resources/data/docs/images/sampler/orthogonal_49_samples_and_proj.svg
   :caption: 49 samples projected onto the first two dimensions and their
             projection on both 1D axis (top and right plot). The pattern is well stratified
             in both 2D and 1D projections. This is true for every pair of dimensions of the
             high-dimentional samples.
.. subfigend::
   :label: fig-orthogonal-pattern

 */

template <typename Float, typename Spectrum>
class OrthogonalSampler final : public RandomSampler<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(RandomSampler, m_sample_count, m_base_seed, m_rng,
                    check_rng, m_samples_per_wavefront, m_wavefront_count)
    MTS_IMPORT_TYPES()

    OrthogonalSampler(const Properties &props = Properties()) : Base(props) {
        m_jitter = props.bool_("jitter", true);
        m_strength = props.int_("strength", 2);

        // Make sure m_resolution is a prime number
        auto is_prime = [](uint32_t x) {
            for (uint32_t i = 2; i <= x / 2; ++i)
                if (x % i == 0)
                    return false;
            return true;
        };

        m_resolution = 2;
        while (sqr(m_resolution) < m_sample_count || !is_prime(m_resolution))
            m_resolution++;

        if (m_sample_count != sqr(m_resolution))
            Log(Warn, "Sample count should be the square of a prime number, rounding to %i", sqr(m_resolution));

        m_sample_count = sqr(m_resolution);

        // Default
        m_samples_per_wavefront = 1;
        m_wavefront_count = m_sample_count;

        m_dimension_index = 0;
        m_wavefront_index = -1;
    }

    ref<Sampler<Float, Spectrum>> clone() override {
        OrthogonalSampler *sampler = new OrthogonalSampler();
        sampler->m_jitter                = m_jitter;
        sampler->m_strength              = m_strength;
        sampler->m_sample_count          = m_sample_count;
        sampler->m_resolution            = m_resolution;
        sampler->m_samples_per_wavefront = m_samples_per_wavefront;
        sampler->m_wavefront_count       = m_wavefront_count;
        sampler->m_base_seed             = m_base_seed;
        sampler->m_dimension_index       = 0;
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

        if (unlikely(m_strength != 2)) {
            return bush(m_wavefront_index + m_wavefront_sample_offsets,
                        m_dimension_index++,
                        m_resolution,
                        m_strength,
                        m_permutations_seed, active);
        } else {
            return bose(m_wavefront_index + m_wavefront_sample_offsets,
                        m_dimension_index++,
                        m_resolution,
                        m_permutations_seed, active);
        }
    }

    Point2f next_2d(Mask active = true) override {
        Assert(m_wavefront_index > -1);
        check_rng(active);

        Float f1 = next_1d(active),
              f2 = next_1d(active);
        return Point2f(f1, f2);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "OrthogonalSampler[" << std::endl
            << "  sample_count = " << m_sample_count << std::endl
            << "  jitter = " << m_jitter << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()




    /// Compute the digits of decimal value ‘i‘ expressed in base ‘b‘
    std::vector<UInt32> to_base_s(UInt32 i, uint32_t b, uint32_t t) {
        std::vector<UInt32> digits(t);
        for (size_t ii = 0; ii < t; i /= b, ++ii)
            digits[ii] = i % b;
        return digits;
    }

    /// Evaluate polynomial with coefficients a at location arg
    UInt32 eval_poly(const std::vector<UInt32> &coef, UInt32 x) {
        UInt32 res = 0;
        for (size_t l = coef.size(); l--; )
            res = (res * x) + coef[l];
        return res;
    }

    /// Bush construction technique for orthogonal arrays
    Float bush(UInt32 i,   // sample index
               uint32_t j, // dimension
               uint32_t s, // number of levels/stratas (has to be a prime number)
               uint32_t t, // strength of OA
               UInt32 p,   // pseudo-random permutation seed
               Mask active = true) {
        uint32_t N = enoki::pow(s, t);
        i = kensler_permute(i, N, p, active);
        auto i_digits = to_base_s(i, s, t);
        uint32_t stm = N / s;
        UInt32 phi = eval_poly(i_digits, j);
        UInt32 stratum = kensler_permute(phi % s, s, p * (j + 1) * 0x51633e2d, active);

        // UInt32 sub_stratum = kensler_permute((i / s) % stm, stm, (i + 1) * p * (j + 1) * 0x68bc21eb, active); // J
        UInt32 sub_stratum = kensler_permute((i / s) % stm, stm, p * (j + 1) * 0x68bc21eb, active); // MJ

        Float jitter = m_jitter ? next_float<Float>(m_rng.get(), active) : 0.5f;
        return (stratum + (sub_stratum + jitter) / stm) / s;
    }

    /// Bose construction technique for orthogonal arrays. It only support OA of strength == 2
    Float bose(UInt32 i,   // sample index
               uint32_t j, // dimension
               uint32_t s, // number of levels/stratas (has to be a prime number)
               UInt32 p,   // pseudo-random permutation seed
               Mask active = true) {

        // Permutes the sample index so that samples are obtained in random order
        i = kensler_permute(i % (s * s), s * s, p, active);

        UInt32 a_i0 = i / s;
        UInt32 a_i1 = i % s;

        UInt32 a_ij, a_ik;
        if (j == 0) {
            a_ij = a_i0;
            a_ik = a_i1;
        } else if (j == 1) {
            a_ij = a_i1;
            a_ik = a_i0;
        } else {
            UInt32 k = (j % 2) ? j - 1 : j + 1;
            a_ij     = (a_i0 + (j - 1) * a_i1) % s;
            a_ik     = (a_i0 + (k - 1) * a_i1) % s;
        }

        UInt32 stratum     = kensler_permute(a_ij, s, p * (j + 1) * 0x51633e2d, active);

        // UInt32 sub_stratum = kensler_permute(a_ik, s, (a_ik * s + a_ij + 1) * p * (j + 1) * 0x68bc21eb, active); // J
        // UInt32 sub_stratum = kensler_permute(a_ik, s, (a_ij + 1) * p * (j + 1) * 0x68bc21eb, active); // MJ
        UInt32 sub_stratum = kensler_permute(a_ik, s, p * (j + 1) * 0x68bc21eb, active); // CMJ

        Float jitter = m_jitter ? next_float<Float>(m_rng.get(), active) : 0.5f;
        return (stratum + (sub_stratum + jitter) / s) / s;
    }

private:
    bool m_jitter;
    ScalarUInt32 m_strength;

    /// Stratification grid resolution
    ScalarUInt32 m_resolution;

    /// Sampler state
    ScalarUInt32 m_dimension_index;
    ScalarInt32  m_wavefront_index;

    UInt32 m_permutations_seed;
    UInt32 m_wavefront_sample_offsets;
};

MTS_IMPLEMENT_CLASS_VARIANT(OrthogonalSampler, Sampler)
MTS_EXPORT_PLUGIN(OrthogonalSampler, "Orthogonal Array Sampler");
NAMESPACE_END(mitsuba)
