// Copyright 2019-2025 Lawrence Livermore National Security, LLC and other YGM
// Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <ygm/container/detail/base_iteration.hpp>
#include <ygm/io/detail/csv.hpp>
#include <ygm/io/line_parser.hpp>

namespace ygm::io {

/**
 * @brief Class for parsing collections of CSV files in distributed memory
 */
class csv_parser : public ygm::container::detail::base_iteration_value<
                       csv_parser, std::tuple<std::vector<detail::csv_field>>> {
 public:
  using for_all_args    = std::tuple<std::vector<detail::csv_field>>;
  using header_map_type = ygm::io::detail::csv_line::header_map_type;

  template <typename... Args>
  csv_parser(Args&&... args)
      : m_lp(std::forward<Args>(args)...), m_has_headers(false) {}

  class iterator {
   public:
    using iterator_category = std::input_iterator_tag;
    using value_type        = ygm::io::detail::csv_field;
    using difference_type   = std::ptrdiff_t;
    using pointer           = const ygm::io::detail::csv_line*;
    using reference         = const ygm::io::detail::csv_line&;

    iterator() = default;  // sentinel/end iterator

    reference operator*() const { return m_impl->m_current_line; }
    pointer   operator->() const { return &m_impl->m_current_line; }

    iterator& operator++() {
      m_impl->advance();
      return *this;
    }

    iterator operator++(int) {
      iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    friend bool operator==(const iterator& a, const iterator& b) {
      bool a_end = !a.m_impl || !a.m_impl->m_lp_iter.valid();
      bool b_end = !b.m_impl || !b.m_impl->m_lp_iter.valid();
      if (a_end || b_end) {
        return a_end == b_end;
      }
      return a.m_impl->m_lp_iter == b.m_impl->m_lp_iter;
    }

    friend bool operator!=(const iterator& a, const iterator& b) {
      return !(a == b);
    }

   private:
    friend class csv_parser;

    struct impl {
      ygm::io::line_parser::iterator m_lp_iter;
      ygm::io::detail::csv_line      m_current_line;
      const header_map_type&         m_header_map;

      impl(const header_map_type& header_map)
          : m_current_line(header_map), m_header_map(header_map) {};

      // Advances to the next line from the underlying line_parser and parses it
      // as a CSV line
      void advance() {
        ++m_lp_iter;
        m_current_line =
            ygm::io::detail::parse_csv_line(*m_lp_iter, m_header_map);
      }
    };

    explicit iterator(std::shared_ptr<impl> impl) : m_impl(std::move(impl)) {}

    std::shared_ptr<impl> m_impl;
  };

  using const_iterator = iterator;

  /**
   * @brief Executes a user function for every CSV record in a set of files.
   *
   * @tparam Function functor type
   * @param fn User function to execute
   */
  template <typename Function>
  void for_all(Function fn) {
    for (const auto& line : *this) {
      fn(line);
    }
  }

  /**
   * @brief Read the header of a CSV file
   */
  void read_headers() {
    using namespace ygm::io::detail;
    auto header_line = m_lp.read_first_line();
    m_lp.set_skip_first_line(true);
    m_header_map  = parse_csv_headers(header_line);
    m_has_headers = true;
  }

  /**
   * @brief Checks for existence of a column label within headers
   *
   * @param label Header label to search for within headers
   */
  bool has_header(const std::string& label) {
    return m_has_headers && (m_header_map.find(label) != m_header_map.end());
  }

  ygm::comm& comm() { return m_lp.comm(); }

  const ygm::comm& comm() const { return m_lp.comm(); }

  /**
   * @brief Returns an iterator to the first line of CSV assigned to this rank
   */
  iterator begin() const {
    auto impl       = std::make_shared<iterator::impl>(m_header_map);
    impl->m_lp_iter = m_lp.begin();
    impl->m_current_line =
        ygm::io::detail::parse_csv_line(*(impl->m_lp_iter), m_header_map);
    return iterator(impl);
  }

  /**
   * @brief Returns a past-the-end sentinel iterator
   */
  iterator end() const { return iterator(); }

 private:
  line_parser m_lp;

  header_map_type m_header_map;
  bool            m_has_headers;
};
}  // namespace ygm::io
