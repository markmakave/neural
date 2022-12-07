#pragma once

#include <cstdlib>

namespace blas {

//
//  Defult allocator similar to std::allocator
//  Uses operator new and operator delete for memory management
//
template <typename T>
class allocator{
        
    allocator() {}
    
public:

    static
    T*
    allocate(size_t size)
    {
        if (size == 0) return nullptr;
        return reinterpret_cast<T*>(operator new(size * sizeof(T)));
    }

    static
    void
    deallocate(T* ptr)
    {
        operator delete(ptr);
    }
};

template <typename T>
class stack_allocator {
        
    stack_allocator() {};
    
public:

    static 
    T*
    allocate(size_t size)
    {
        if (size == 0) return nullptr;
        return alloca(size * sizeof(T));
    }

    static 
    void 
    deallocate(T* ptr) {}

};

template <typename T1, typename T2>
void
memcpy(T1* dest, const T2* src, size_t size)
{
    for (size_t i = 0; i < size; ++i)
    {
        *dest++ = *src++;
    }
}

template <typename T>
void
swap(T& x, T& y)
{
    const T temp = x;
    x = y;
    y = x;
}

}
