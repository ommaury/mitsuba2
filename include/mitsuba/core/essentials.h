#pragma once

#include <mitsuba/core/platform.h>
#include <cstdint>
#include <cstring>
#include <utility>

NAMESPACE_BEGIN(mitsuba)

/// Cast between types that have an identical binary representation.
template<typename T, typename U> T memcpy_cast(const U &val) {
	static_assert(sizeof(T) == sizeof(U), "memcpy_cast: sizes did not match!");
	T result;
    std::memcpy(&result, &val, sizeof(T));
    return result;
}

/** \brief The following is used to ensure that the getters and setters
 * for all the same types are available for both \ref Stream implementations
 * and \AnnotatedStream. */

template<typename... Args> struct for_each_type;

template <typename T, typename... Args>
struct for_each_type<T, Args...> {
    template <typename UserFunctionType, typename ...Params>
    static void recurse(Params&&... params) {
        UserFunctionType::template apply<T>(std::forward<Params>(params)...);
        for_each_type<Args...>::template recurse<UserFunctionType>(std::forward<Params>(params)...);
    }
};

/// Base case
template<>
struct for_each_type<> {
    template <typename UserFunctionType, typename... Params>
    static void recurse(Params&&...) { }
};

NAMESPACE_END(mitsuba)